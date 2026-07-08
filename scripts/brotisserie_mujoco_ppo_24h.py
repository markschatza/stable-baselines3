#!/usr/bin/env python3
"""Time-budgeted SB3 PPO baseline for classic Gymnasium MuJoCo.

This is the fallback MuJoCo path for Brotisserie when MuJoCo Playground/MJX
training is not stable on AMD ROCm. Simulation is CPU-side Gymnasium/MuJoCo;
policy/value training uses PyTorch ROCm via SB3's `cuda` device.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class JsonlCallback(BaseCallback):
    def __init__(self, path: Path, log_every_steps: int = 10_000):
        super().__init__()
        self.path = path
        self.log_every_steps = log_every_steps
        self.last_log_step = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step < self.log_every_steps:
            return True
        self.last_log_step = self.num_timesteps
        row = {
            "wall_time_s": time.time() - self.start_time,
            "num_timesteps": self.num_timesteps,
        }
        if torch.cuda.is_available():
            row.update(
                {
                    "torch_cuda_device": torch.cuda.get_device_name(0),
                    "torch_cuda_mem_allocated": torch.cuda.memory_allocated(0),
                    "torch_cuda_mem_reserved": torch.cuda.memory_reserved(0),
                }
            )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return True


def record_video(model: PPO, env_id: str, path: Path, max_steps: int, seed: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    env.close()
    if frames:
        imageio.mimsave(path, frames, fps=30)
    return {
        "video_path": str(path),
        "video_bytes": path.stat().st_size if path.exists() else 0,
        "episode_reward": total_reward,
        "episode_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v5")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--video-interval-s", type=float, default=3600.0)
    parser.add_argument("--initial-chunk-steps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    metadata = {
        "env": args.env,
        "hours": args.hours,
        "video_interval_s": args.video_interval_s,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "note": "Gymnasium/MuJoCo CPU simulation + SB3 PPO PyTorch policy/value training on requested device.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_env = make_vec_env(
        args.env,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(run_dir / "monitor"),
    )
    eval_env = Monitor(gym.make(args.env))

    model = PPO(
        "MlpPolicy",
        train_env,
        device=args.device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
    )

    metrics_path = run_dir / "metrics.jsonl"
    video_manifest_path = run_dir / "video_manifest.jsonl"
    callback = JsonlCallback(metrics_path)

    deadline = time.time() + args.hours * 3600.0
    next_video_time = time.time()
    chunk_steps = args.initial_chunk_steps
    chunk_idx = 0
    start = time.time()

    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as events:
        while time.time() < deadline:
            chunk_idx += 1
            before_steps = model.num_timesteps
            before_time = time.time()
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, callback=callback)
            elapsed = time.time() - before_time
            steps_done = model.num_timesteps - before_steps
            sps = steps_done / elapsed if elapsed > 0 else 0.0

            ckpt = run_dir / "checkpoints" / f"step_{model.num_timesteps}.zip"
            model.save(ckpt)

            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
            )
            row = {
                "type": "chunk_complete",
                "chunk_idx": chunk_idx,
                "wall_time_s": time.time() - start,
                "num_timesteps": model.num_timesteps,
                "chunk_steps": steps_done,
                "chunk_elapsed_s": elapsed,
                "chunk_sps": sps,
                "eval_mean_reward": float(mean_reward),
                "eval_std_reward": float(std_reward),
                "checkpoint": str(ckpt),
            }
            events.write(json.dumps(row) + "\n")
            events.flush()

            if time.time() >= next_video_time:
                video_path = run_dir / "videos" / f"step_{model.num_timesteps}.mp4"
                video = record_video(
                    model,
                    args.env,
                    video_path,
                    max_steps=args.video_max_steps,
                    seed=args.seed + chunk_idx,
                )
                video_row = {**row, "type": "video", **video}
                with video_manifest_path.open("a", encoding="utf-8") as vf:
                    vf.write(json.dumps(video_row) + "\n")
                next_video_time += args.video_interval_s

            # Adapt chunk size toward the requested video interval so videos land roughly hourly.
            if sps > 0:
                chunk_steps = max(args.n_envs * args.n_steps, int(sps * args.video_interval_s))

    final_path = run_dir / "final_model.zip"
    model.save(final_path)
    train_env.close()
    eval_env.close()
    print(json.dumps({"final_model": str(final_path), "timesteps": model.num_timesteps}))


if __name__ == "__main__":
    main()
