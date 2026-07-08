#!/usr/bin/env python3
"""Single-device JAX PPO for MuJoCo Playground/MJX on AMD ROCm.

This intentionally avoids Brax PPO's pmap/multi-device machinery, which has
been unstable on the local RX 7900 GRE ROCm stack. It is a compact baseline PPO
runner for GPU-resident MJX env stepping + policy/value updates.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax.linen as nn
from flax.training.train_state import TrainState
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
from mujoco_playground import registry
from mujoco_playground import wrapper
import numpy as np
import optax

Array = jax.Array


@dataclass(frozen=True)
class Config:
    env_name: str = "G1JoystickRoughTerrain"
    total_timesteps: int = 1_000_000
    num_envs: int = 512
    num_steps: int = 64
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.97
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 0
    eval_every_updates: int = 10
    video_every_updates: int = 10
    checkpoint_every_updates: int = 10
    video_max_steps: int = 120
    episode_length: int = 1000
    action_repeat: int = 1
    hidden_sizes: tuple[int, ...] = (512, 256, 128)


class ActorCritic(nn.Module):
    action_dim: int
    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        h = x
        for width in self.hidden_sizes:
            h = nn.tanh(nn.Dense(width)(h))
        mean = nn.Dense(self.action_dim)(h)
        value = nn.Dense(1)(h).squeeze(-1)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, log_std, value


def obs_array(obs: Any) -> Array:
    if isinstance(obs, dict):
        return obs["state"]
    return obs


def normal_log_prob(action: Array, mean: Array, log_std: Array) -> Array:
    std = jnp.exp(log_std)
    logp = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    return logp.sum(axis=-1)


def normal_entropy(log_std: Array) -> Array:
    return (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).sum(axis=-1)


def make_train_fns(env: Any, model: ActorCritic, cfg: Config):
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate))

    def create_state(rng: Array, obs_dim: int) -> TrainState:
        params = model.init(rng, jnp.zeros((1, obs_dim)))
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def reset_env(rng: Array):
        return env.reset(jax.random.split(rng, cfg.num_envs))

    def rollout_step(carry, _):
        train_state, env_state, rng = carry
        rng, action_key = jax.random.split(rng)
        obs = obs_array(env_state.obs)
        mean, log_std, value = train_state.apply_fn(train_state.params, obs)
        noise = jax.random.normal(action_key, mean.shape)
        action = mean + jnp.exp(log_std) * noise
        clipped_action = jnp.clip(action, -1.0, 1.0)
        logp = normal_log_prob(action, mean, log_std)
        next_state = env.step(env_state, clipped_action)
        transition = {
            "obs": obs,
            "action": action,
            "logp": logp,
            "reward": next_state.reward,
            "done": next_state.done,
            "value": value,
        }
        return (train_state, next_state, rng), transition

    @jax.jit
    def collect_rollout(train_state: TrainState, env_state: Any, rng: Array):
        (train_state, env_state, rng), traj = jax.lax.scan(
            rollout_step, (train_state, env_state, rng), None, length=cfg.num_steps
        )
        last_obs = obs_array(env_state.obs)
        _, _, last_value = train_state.apply_fn(train_state.params, last_obs)
        return env_state, rng, traj, last_value

    @jax.jit
    def compute_gae(traj: dict[str, Array], last_value: Array):
        def gae_step(carry, transition):
            next_gae, next_value = carry
            delta = transition["reward"] + cfg.gamma * next_value * (1.0 - transition["done"]) - transition["value"]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - transition["done"]) * next_gae
            return (gae, transition["value"]), gae

        _, advantages_rev = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(last_value), last_value),
            jax.tree.map(lambda x: x[::-1], traj),
        )
        advantages = advantages_rev[::-1]
        returns = advantages + traj["value"]
        return advantages, returns

    def loss_fn(params: Any, batch: dict[str, Array]):
        mean, log_std, value = model.apply(params, batch["obs"])
        new_logp = normal_log_prob(batch["action"], mean, log_std)
        logratio = new_logp - batch["logp"]
        ratio = jnp.exp(logratio)
        advantages = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-8)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((value - batch["return"]) ** 2).mean()
        entropy = normal_entropy(log_std).mean()
        loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        metrics = {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1) - logratio).mean(),
            "clip_frac": (jnp.abs(ratio - 1.0) > cfg.clip_coef).mean(),
        }
        return loss, metrics

    @jax.jit
    def update(train_state: TrainState, traj: dict[str, Array], advantages: Array, returns: Array, rng: Array):
        batch = {**traj, "advantage": advantages, "return": returns}
        batch = jax.tree.map(lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), batch)
        batch_size = cfg.num_steps * cfg.num_envs
        minibatch_size = batch_size // cfg.num_minibatches

        def epoch_step(carry, _):
            train_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, batch_size)
            shuffled = jax.tree.map(lambda x: x[perm], batch)
            minibatches = jax.tree.map(
                lambda x: x.reshape((cfg.num_minibatches, minibatch_size, *x.shape[1:])), shuffled
            )

            def minibatch_step(ts: TrainState, mb: dict[str, Array]):
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params, mb)
                ts = ts.apply_gradients(grads=grads)
                return ts, metrics

            train_state, metrics = jax.lax.scan(minibatch_step, train_state, minibatches)
            metrics = jax.tree.map(jnp.mean, metrics)
            return (train_state, rng), metrics

        (train_state, rng), metrics = jax.lax.scan(epoch_step, (train_state, rng), None, cfg.update_epochs)
        return train_state, rng, jax.tree.map(jnp.mean, metrics)

    return create_state, reset_env, collect_rollout, compute_gae, update


def render_video(env_name: str, params: Any, model: ActorCritic, cfg: Config, path: Path, seed: int) -> dict[str, Any]:
    eval_env = registry.load(env_name, config=registry.get_default_config(env_name), config_overrides={"impl": "jax"})
    state = jax.jit(eval_env.reset)(jax.random.PRNGKey(seed))
    jit_step = jax.jit(eval_env.step)
    rollout = []
    total_reward = 0.0
    done = False
    for _ in range(min(cfg.episode_length, cfg.video_max_steps)):
        obs = obs_array(state.obs)[None, ...]
        mean, _, _ = model.apply(params, obs)
        action = jnp.clip(mean[0], -1.0, 1.0)
        state = jit_step(state, action)
        rollout.append(state)
        total_reward += float(state.reward)
        done = bool(state.done)
        if done:
            break
    frames = eval_env.render(rollout[::2], height=480, width=640, scene_option=mujoco.MjvOption())
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=max(1, int(1 / eval_env.dt / 2)))
    return {
        "video_path": str(path),
        "video_bytes": path.stat().st_size,
        "episode_reward": total_reward,
        "episode_steps": len(rollout),
    }


def scalarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-name", default="G1JoystickRoughTerrain")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every-updates", type=int, default=10)
    p.add_argument("--video-every-updates", type=int, default=10)
    p.add_argument("--checkpoint-every-updates", type=int, default=10)
    p.add_argument("--video-max-steps", type=int, default=120)
    args = p.parse_args()

    cfg = Config(
        env_name=args.env_name,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        seed=args.seed,
        eval_every_updates=args.eval_every_updates,
        video_every_updates=args.video_every_updates,
        checkpoint_every_updates=args.checkpoint_every_updates,
        video_max_steps=args.video_max_steps,
    )
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    raw_env = registry.load(cfg.env_name, config=registry.get_default_config(cfg.env_name), config_overrides={"impl": "jax"})
    env = wrapper.wrap_for_brax_training(raw_env, episode_length=cfg.episode_length, action_repeat=cfg.action_repeat)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_key, reset_key = jax.random.split(rng, 3)
    create_state, reset_env, collect_rollout, compute_gae, update = make_train_fns(
        env, ActorCritic(raw_env.action_size, cfg.hidden_sizes), cfg
    )
    env_state = reset_env(reset_key)
    sample_obs = obs_array(env_state.obs)
    train_state = create_state(init_key, sample_obs.shape[-1])
    model = ActorCritic(raw_env.action_size, cfg.hidden_sizes)

    total_updates = math.ceil(cfg.total_timesteps / (cfg.num_envs * cfg.num_steps))
    start = time.time()
    event_path = run_dir / "events.jsonl"
    for update_idx in range(1, total_updates + 1):
        t0 = time.time()
        env_state, rng, traj, last_value = collect_rollout(train_state, env_state, rng)
        advantages, returns = compute_gae(traj, last_value)
        train_state, rng, metrics = update(train_state, traj, advantages, returns, rng)
        jax.tree.map(lambda x: x.block_until_ready(), metrics)
        steps = update_idx * cfg.num_envs * cfg.num_steps
        row = {
            "update": update_idx,
            "timesteps": steps,
            "wall_time_s": time.time() - start,
            "sps": cfg.num_envs * cfg.num_steps / max(time.time() - t0, 1e-9),
            "mean_reward": float(np.asarray(traj["reward"]).mean()),
            **scalarize_metrics(metrics),
        }
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if update_idx % cfg.checkpoint_every_updates == 0 or update_idx == total_updates:
            ckpt = run_dir / "checkpoints" / f"step_{steps}.npz"
            flat, treedef = jax.tree.flatten(jax.device_get(train_state.params))
            np.savez(ckpt, *flat, treedef=repr(treedef))
        if update_idx % cfg.video_every_updates == 0 or update_idx == total_updates:
            video = render_video(
                cfg.env_name,
                train_state.params,
                model,
                cfg,
                run_dir / "videos" / f"step_{steps}.mp4",
                cfg.seed + update_idx,
            )
            with (run_dir / "video_manifest.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, **video}) + "\n")


if __name__ == "__main__":
    main()
