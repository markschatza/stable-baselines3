# Brotisserie Hard PPO Benchmark Suite

The smoke envs were useful for plumbing, but they are too easy to expose meaningful PPO improvements. The main benchmark suite should use trainings that take hours and have real failure modes.

## Principle

Use environments where PPO has to manage at least one of:

- long-horizon continuous control,
- high-variance locomotion,
- visual CNN policies,
- sparse rewards / partial observability,
- enough wall-clock duration that optimizer stability and throughput matter.

We should keep quick smoke tests, but not optimize against them.

## Local dependency status

Installed into `.venv`:

```bash
uv pip install 'gymnasium[mujoco,box2d,atari,accept-rom-license]' minigrid
```

Observed caveat: Gymnasium 1.3.0 does not expose an `accept-rom-license` extra, but `ale-py` is installed and Atari envs register after importing/registering `ale_py`.

Verified env creation locally:

- `BipedalWalker-v3`
- `LunarLander-v3`
- `HalfCheetah-v5`
- `Hopper-v5`
- `Walker2d-v5`
- `Ant-v5`
- `Humanoid-v5`
- `CarRacing-v3`
- `MiniGrid-FourRooms-v0` after importing `minigrid`
- `PongNoFrameskip-v4` / `ALE/Pong-v5` after registering `ale_py`
- `BreakoutNoFrameskip-v4` / `ALE/Breakout-v5` after registering `ale_py`

## Suite file

Machine-readable suite:

```text
references/brotisserie_hard_ppo_suite.json
```

## Recommended tiers

### Tier A: Box2D control, hours but still convenient

Good first real benchmark because install friction is low and PPO has meaningful room to improve.

| Env | Timesteps | Why |
|---|---:|---|
| `BipedalWalker-v3` | 5M | Classic PPO locomotion stressor; tuned target around 288 in RL Zoo public table. |
| `LunarLander-v3` | 1M | Discrete control with shaped reward; RL Zoo has current v3 tuned config. |

### Tier B: MuJoCo locomotion, main default-PPO battleground

These are the strongest candidates for testing a better default PPO recipe.

| Env | Timesteps | Notes |
|---|---:|---|
| `HalfCheetah-v5` | 1M | Good sample-efficiency benchmark; proxy RL Zoo config exists for v4. |
| `Hopper-v5` | 1M | More brittle than HalfCheetah; useful for stability. |
| `Walker2d-v5` | 1M | Longer-horizon gait; PPO defaults often show meaningful variance. |
| `Ant-v5` | 1M | Higher-dimensional action/obs; useful robustness test. |
| `Humanoid-v5` | 10M | Heavyweight candidate; use after harness is solid. |

### Tier C: Visual GPU suite

This is where the RX 7900 GRE should matter more.

| Env | Timesteps | Policy | Notes |
|---|---:|---|---|
| `CarRacing-v3` | 4M | `CnnPolicy` | Local visual continuous-control task; RL Zoo has a v3 tuned config. |
| `PongNoFrameskip-v4` | 10M | `CnnPolicy` | Public RL Zoo PPO target: ~20.99. |
| `BreakoutNoFrameskip-v4` | 10M | `CnnPolicy` | Public RL Zoo PPO target: ~398.03. |
| `SeaquestNoFrameskip-v4` | 10M | `CnnPolicy` | Harder Atari target; good for throughput/stability. |

### Tier D: Sparse exploration / failure modes

Useful once we care about default behavior beyond dense-control tasks.

| Env | Timesteps | Notes |
|---|---:|---|
| `MiniGrid-FourRooms-v0` | 5M | Public RL Zoo PPO target: ~0.573. |
| `MiniGrid-LockedRoom-v0` | 10M | RL Zoo marks as unsolved; good failure-mode test. |
| `MiniGrid-ObstructedMaze-2Dlh-v0` | 10M | RL Zoo marks as unsolved; good failure-mode test. |

## First long-running command candidates

### CPU/MuJoCo baseline batch

For MLP MuJoCo, CPU may beat GPU despite the AMD GPU being available because environment stepping dominates.

```bash
cd /home/marks/c/git/stable-baselines3
. .venv/bin/activate
HIP_VISIBLE_DEVICES=0 python scripts/brotisserie_baseline_ppo.py \
  --device cpu \
  --policy MlpPolicy \
  --timesteps 1000000 \
  --eval-episodes 20 \
  --env HalfCheetah-v5 \
  --env Hopper-v5 \
  --env Walker2d-v5 \
  --env Ant-v5 \
  --seed 0 \
  --out reports/brotisserie_baselines/ppo_default_mujoco_v5_1m.jsonl
```

### Single heavyweight MuJoCo candidate

```bash
HIP_VISIBLE_DEVICES=0 python scripts/brotisserie_baseline_ppo.py \
  --device cpu \
  --policy MlpPolicy \
  --timesteps 10000000 \
  --eval-episodes 20 \
  --env Humanoid-v5 \
  --seed 0 \
  --out reports/brotisserie_baselines/ppo_default_humanoid_v5_10m.jsonl
```

### Visual GPU candidate

```bash
HIP_VISIBLE_DEVICES=0 python scripts/brotisserie_baseline_ppo.py \
  --device cuda \
  --policy CnnPolicy \
  --timesteps 4000000 \
  --eval-episodes 20 \
  --env CarRacing-v3 \
  --seed 0 \
  --out reports/brotisserie_baselines/ppo_default_carracing_v3_4m_gpu.jsonl
```

## Benchmark policy going forward

- Smoke tests stay tiny and only validate plumbing.
- Brotisserie PPO improvements should be judged on Tier A/B/C, not CartPole/Pendulum.
- For each hard env, compare:
  1. `sb3_default`: local upstream SB3 PPO defaults.
  2. `rl_zoo_tuned`: local run using RL Zoo tuned config or closest versioned proxy.
  3. `brotisserie_default`: our candidate default/heuristic PPO.
  4. `rl_zoo_public_target`: public one-run trained-agent number, clearly labeled as a target/sanity check.
- Report wall-clock, steps/sec, eval reward, variance across seeds, and PPO health metrics once we add them.
