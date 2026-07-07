"""Run small Brotisserie PPO baseline sweeps.

This is intentionally lightweight: it uses upstream SB3 PPO defaults unless a CLI
flag overrides them, evaluates the learned policy, and writes machine-readable
JSONL results. It is meant for quick local smoke/baseline runs before larger RL
Zoo comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        action="append",
        dest="envs",
        default=None,
        help="Gymnasium environment id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        default=None,
        help="Seed. Can be passed multiple times.",
    )
    parser.add_argument("--timesteps", type=int, default=10_000, help="Training timesteps per run.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per run.")
    parser.add_argument("--device", default="auto", help="SB3/PyTorch device, e.g. auto, cpu, cuda.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/brotisserie_baselines/ppo_default_smoke.jsonl"),
        help="JSONL output path.",
    )
    return parser.parse_args()


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": th.__version__,
        "torch_hip": getattr(th.version, "hip", None),
        "torch_cuda_available": th.cuda.is_available(),
        "torch_cuda_device_count": th.cuda.device_count(),
    }
    if th.cuda.is_available():
        info["torch_cuda_device_name_0"] = th.cuda.get_device_name(0)
    return info


def make_monitored_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    return Monitor(env)


def run_one(env_id: str, seed: int, timesteps: int, eval_episodes: int, device: str) -> dict[str, Any]:
    train_env = make_monitored_env(env_id, seed)
    eval_env = make_monitored_env(env_id, seed + 10_000)

    start = time.perf_counter()
    model = PPO("MlpPolicy", train_env, seed=seed, device=device, verbose=0)
    resolved_device = str(model.device)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_seconds = time.perf_counter() - start

    eval_result = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        warn=False,
    )
    mean_reward, std_reward = cast(tuple[float, float], eval_result)

    result: dict[str, Any] = {
        "algo": "PPO",
        "variant": "sb3_default",
        "env_id": env_id,
        "seed": seed,
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "train_seconds": train_seconds,
        "steps_per_second": timesteps / train_seconds,
        "requested_device": device,
        "resolved_device": resolved_device,
        **torch_info(),
    }

    train_env.close()
    eval_env.close()
    return result


def main() -> None:
    args = parse_args()
    envs = args.envs or ["CartPole-v1", "Pendulum-v1"]
    seeds = args.seeds or [0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as file_handler:
        for env_id in envs:
            for seed in seeds:
                result = run_one(env_id, seed, args.timesteps, args.eval_episodes, args.device)
                print(json.dumps(result, sort_keys=True))
                file_handler.write(json.dumps(result, sort_keys=True) + "\n")
                file_handler.flush()


if __name__ == "__main__":
    main()
