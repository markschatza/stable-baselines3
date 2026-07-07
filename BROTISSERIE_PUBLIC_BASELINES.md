# Public PPO Defaults and Benchmarks

This file makes the public baseline material visible inside the Brotisserie SB3 fork.

## Sources grabbed

| Source | What we use | Local copy |
|---|---|---|
| Stable-Baselines3 | Upstream PPO constructor defaults | live introspection from this checkout |
| RL Baselines3 Zoo | Tuned PPO hyperparameters | `references/public_baselines/rl_zoo_ppo.yml` |
| RL Baselines3 Zoo | Selected tuned PPO config blocks | `references/public_baselines/rl_zoo_ppo_selected.yml` |
| RL Baselines3 Zoo | Public trained-agent benchmark rows | `references/public_baselines/rl_zoo_ppo_benchmark_selected.md` |
| Hugging Face SB3 | Model cards, videos, trained agents | https://huggingface.co/sb3 |
| OpenRL Benchmark / W&B | SB3 benchmark dashboard link from SB3 README | https://wandb.ai/openrlbenchmark/sb3 |

Source revisions at extraction time:

- This SB3 fork commit: `bf08f3db7e18e87b68adbf90c1851e5e244cb9df`
- Local `rl-baselines3-zoo` commit: `ecfecc9ed2460c46e8357413a471bcb2b7600087`

Important caveat from RL Zoo: its `benchmark.md` table is not a rigorous multi-seed benchmark. It is a one-run
trained-agent performance table intended to check maximal algorithm performance, find bugs, and make pretrained agents
available.

## SB3 PPO constructor defaults

These are the upstream SB3 defaults from `stable_baselines3.PPO.__init__` in this checkout.

| parameter | SB3 default |
|---|---:|
| `learning_rate` | `0.0003` |
| `n_steps` | `2048` |
| `batch_size` | `64` |
| `n_epochs` | `10` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `clip_range_vf` | `None` |
| `normalize_advantage` | `True` |
| `ent_coef` | `0.0` |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `use_sde` | `False` |
| `sde_sample_freq` | `-1` |
| `target_kl` | `None` |
| `stats_window_size` | `100` |
| `policy_kwargs` | `None` |

## RL Zoo tuned config coverage we care about first

The selected local YAML includes `default`, `atari`, and any matching entries for our candidate benchmark environments.

```text
references/public_baselines/rl_zoo_ppo_selected.yml
```

Key available selected entries currently include:

- `default`
- `atari`
- `CartPole-v1`
- `Pendulum-v1`
- `MountainCarContinuous-v0`
- `Acrobot-v1`
- `LunarLander-v3`
- `BipedalWalker-v3`

## Selected RL Zoo PPO benchmark rows

See:

```text
references/public_baselines/rl_zoo_ppo_benchmark_selected.md
```

Those rows give us public tuned-agent target numbers for common envs such as CartPole, Pendulum, BipedalWalker,
classic MuJoCo v3 tasks, and Atari.

## How this fits Brotisserie

We should expose three columns in future result tables:

1. `sb3_default`: our local run of upstream SB3 PPO defaults.
2. `rl_zoo_tuned`: local run using RL Zoo PPO tuned hyperparameters where available.
3. `rl_zoo_public_target`: public RL Zoo trained-agent score from `benchmark.md`, clearly labeled as a one-run public
   target, not a statistically rigorous benchmark.

This gives us both local reproducibility and a visible public target for sanity checking.

## Refresh command

```bash
cd /home/marks/c/git/stable-baselines3
. .venv/bin/activate
python scripts/brotisserie_public_baselines.py --rl-zoo ../rl-baselines3-zoo
```
