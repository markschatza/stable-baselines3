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
from stable_baselines3.common.env_util import make_vec_env
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
    parser.add_argument("--policy", default="MlpPolicy", help="SB3 policy name, e.g. MlpPolicy or CnnPolicy.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of vectorized training environments.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override PPO n_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPO batch_size.")
    parser.add_argument("--n-epochs", type=int, default=None, help="Override PPO n_epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override PPO learning_rate.")
    parser.add_argument("--tensorboard-log", type=Path, default=None, help="Optional TensorBoard log directory.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/brotisserie_baselines/ppo_default_smoke.jsonl"),
        help="JSONL output path.",
    )
    return parser.parse_args()


def register_optional_envs() -> None:
    """Register optional env packages when installed.

    Gymnasium 1.x exposes Atari via ale-py only after explicit registration, and
    MiniGrid registers env ids on import. Import failures are fine: dependency
    availability is checked by the env creation path.
    """
    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError:
        pass

    try:
        import minigrid  # noqa: F401
    except ImportError:
        pass


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
    if env_id.startswith("MiniGrid-"):
        from minigrid.wrappers import FlatObsWrapper

        env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return Monitor(env)


def run_one(args: argparse.Namespace, env_id: str, seed: int) -> dict[str, Any]:
    ppo_kwargs: dict[str, Any] = {}
    for cli_name, ppo_name in [
        ("n_steps", "n_steps"),
        ("batch_size", "batch_size"),
        ("n_epochs", "n_epochs"),
        ("learning_rate", "learning_rate"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            ppo_kwargs[ppo_name] = value

    train_env = make_vec_env(lambda: make_monitored_env(env_id, seed), n_envs=args.n_envs, seed=seed)
    eval_env = make_monitored_env(env_id, seed + 10_000)

    start = time.perf_counter()
    model = PPO(
        args.policy,
        train_env,
        seed=seed,
        device=args.device,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log is not None else None,
        verbose=0,
        **ppo_kwargs,
    )
    resolved_device = str(model.device)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_seconds = time.perf_counter() - start

    eval_result = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        warn=False,
    )
    mean_reward, std_reward = cast(tuple[float, float], eval_result)

    result: dict[str, Any] = {
        "algo": "PPO",
        "variant": "sb3_default",
        "env_id": env_id,
        "seed": seed,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "train_seconds": train_seconds,
        "steps_per_second": args.timesteps / train_seconds,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "policy": args.policy,
        "n_envs": args.n_envs,
        "ppo_overrides": ppo_kwargs,
        **torch_info(),
    }

    train_env.close()
    eval_env.close()
    return result


def main() -> None:
    register_optional_envs()
    args = parse_args()
    envs = args.envs or ["CartPole-v1", "Pendulum-v1"]
    seeds = args.seeds or [0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as file_handler:
        for env_id in envs:
            for seed in seeds:
                result = run_one(args, env_id, seed)
                print(json.dumps(result, sort_keys=True))
                file_handler.write(json.dumps(result, sort_keys=True) + "\n")
                file_handler.flush()


if __name__ == "__main__":
    main()
