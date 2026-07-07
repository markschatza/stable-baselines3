# Brotisserie 24h GPU-Bound Benchmark Direction

This supersedes the earlier CPU-heavy MuJoCo/SB3 benchmark emphasis. The target is now **GPU-bound training that can occupy the RX 7900 GRE for many hours**, not small SB3/Gymnasium envs where CPU simulation dominates.

## Key correction

If the environment can run mostly on CPU and only occasionally update a small MLP on GPU, it is probably not a good primary benchmark for Brotisserie PPO.

The main benchmark should be environments where either:

1. the **environment itself is GPU/JIT accelerated**, or
2. the observation/model workload is large enough that the **policy update is materially GPU-bound**, or
3. both are true.

That means the strongest track is probably **JAX/ROCm**, not SB3/PyTorch-only.

## Local ROCm/JAX status

Created a separate ROCm JAX uv env so we do not disturb the SB3/PyTorch env:

```bash
cd /home/marks/c/git/stable-baselines3
uv venv --python python3.12 .venv-jax-rocm
. .venv-jax-rocm/bin/activate
uv pip install pip
uv pip install 'jax==0.10.2' 'jax-rocm7-pjrt==0.10.2' 'jax-rocm7-plugin==0.10.2'
```

Verified on the local AMD Radeon RX 7900 GRE:

```text
jax 0.10.2
devices [RocmDevice(id=0)]
backend gpu
matmul 1024.0
```

So JAX on ROCm works on this machine.

## Craftax status

Installed and smoke-tested Craftax in `.venv-jax-rocm`:

```bash
. .venv-jax-rocm/bin/activate
uv pip install craftax
HIP_VISIBLE_DEVICES=0 python - <<'PY'
import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
print('backend', jax.default_backend(), jax.devices())
env = make_craftax_env_from_name('Craftax-Symbolic-v1', auto_reset=True)
rng = jax.random.PRNGKey(0)
params = env.default_params
obs, state = env.reset(rng, params)
print('obs shape', getattr(obs, 'shape', None))
obs, state, reward, done, info = jax.jit(env.step)(rng, state, jnp.array(0), params)
jax.block_until_ready(reward)
print(float(reward), bool(done))
PY
```

Observed:

```text
backend gpu [RocmDevice(id=0)]
obs shape (8268,)
step ok 0.0 False
```

This is now the best first candidate for a true GPU-bound PPO investigation.

## Candidate benchmark families

### 1. Craftax / Craftax-Classic — strongest first target

Why it fits:

- Written entirely in JAX.
- Runs env stepping and PPO pipeline on GPU when using JAX backend.
- Has long-horizon sparse achievement/reward structure.
- Public baselines exist.
- PPO/RNN/RND/ICM/E3B variants exist in `Craftax_Baselines`.
- We already verified Craftax steps on ROCm GPU locally.

Public reference points from Craftax README:

#### Craftax-1B

| Method | Score |
|---|---:|
| PPO-GTrXL | 18.3 |
| PQN-RNN | 16.0 |
| PPO-RNN | 15.3 |
| RND | 12.0 |
| PPO | 11.9 |
| ICM | 11.9 |
| E3B | 11.0 |

#### Craftax-1M

| Method | Score |
|---|---:|
| PPO-RNN | 2.3 |
| RND | 2.2 |
| PPO | 2.2 |
| ICM | 2.2 |
| E3B | 2.2 |

Local clone:

```text
/home/marks/c/git/Craftax_Baselines
```

Initial command reference from upstream:

```bash
python ppo.py
python ppo_rnn.py
python ppo.py --train_icm
python ppo_rnd.py
```

Next action should be adapting/running Craftax_Baselines under `.venv-jax-rocm` on the 7900 GRE and logging throughput, VRAM, score, and wall-clock.

### 2. MuJoCo Playground / MJX robot sims — best hard robotics target if ROCm JAX cooperates

Why it fits:

- GPU-accelerated robot learning built with MuJoCo MJX.
- JAX PPO training scripts exist.
- Includes locomotion/manipulation/vision environments.
- Much closer to “hard robot sims” than classic Gym MuJoCo.

Caveat:

- Official setup docs emphasize CUDA JAX, but because JAX ROCm works locally, we should test source install against `.venv-jax-rocm`.
- This is a better AMD-compatible robotics bet than Isaac Lab.

Candidate tasks to investigate:

- `PandaPickCube`
- `G1JoystickFlatTerrain`
- quadruped / humanoid locomotion tasks from MuJoCo Playground
- vision variants if they run on ROCm without CUDA-only pieces

### 3. Brax / MJX

Why it fits:

- JAX-native physics/training.
- Designed for accelerator-scale environment stepping.
- Good baseline infrastructure for PPO.

Caveat:

- Brax docs currently suggest MuJoCo Playground for newer robot environments.
- Good fallback if MuJoCo Playground install is too heavy.

### 4. Procgen

Procgen is still useful, but it does **not** fully match the new GPU-bound requirement.

Why:

- Procgen environments are CPU-generated and extremely fast on CPU.
- The policy is CNN-based and can use GPU, but the environment itself is not GPU-native.
- It is still valuable for generalization and visual-control experiments, but should not be the primary 24h GPU-saturation benchmark.

If used, prefer a large multi-env CNN setup with many Procgen envs and long training, but classify it as **visual generalization**, not true GPU-sim.

### 5. Isaac Lab / Isaac Sim

Not a good fit for this AMD 7900 GRE machine.

Why:

- Isaac Lab is GPU-accelerated and robotics-heavy, but it is built on NVIDIA Isaac Sim.
- Isaac Sim/RTX/CUDA expectations make AMD ROCm support unlikely to be practical.

### 6. Genesis World

Promising but higher-risk.

Why it might fit:

- Claims compiler backends including AMD ROCm.
- Robotics/multi-physics direction is relevant.

Caveats:

- RL benchmark maturity and AMD path need verification.
- Treat as exploratory after Craftax and MJX.

## Revised priority order

1. **Craftax_Baselines PPO on JAX ROCm** — immediate next step.
2. **Craftax PPO-RNN / RND / ICM variants** — hard exploration/default robustness tests.
3. **MuJoCo Playground MJX PPO on JAX ROCm** — hard robot sim track.
4. **Brax/MJX PPO** — fallback accelerator-native robot/control track.
5. **Procgen CNN PPO** — visual generalization track, not primary GPU-sim track.
6. **Genesis World** — exploratory AMD robot sim track.
7. **Isaac Lab** — deprioritize on AMD.

## What this means for SB3 fork

SB3 remains useful as:

- a familiar PPO reference implementation,
- a baseline API/design comparison,
- a place to document our baseline investigation,
- potentially a PyTorch implementation target later.

But for true 24h GPU-bound env training on AMD, the main experimental harness should likely live beside this fork and use JAX/ROCm first.

The next coding task should be a `brotisserie-craftax` runner that:

- uses `.venv-jax-rocm`,
- runs Craftax PPO from `Craftax_Baselines`,
- logs config, score, throughput, wall-clock, and GPU memory/utilization,
- supports 1M / 100M / 1B step presets,
- can run unattended for 24h with checkpointing.
