# Brotisserie GPU-Native MuJoCo PPO Fix

## Result

Built and verified a **single-device JAX PPO runner for MuJoCo Playground/MJX** that avoids the Brax PPO `pmap` path that segfaults on the local AMD ROCm stack.

New runner:

```text
scripts/brotisserie_mjx_single_device_ppo.py
```

This is now the working GPU-native MuJoCo path on the RX 7900 GRE:

```text
MuJoCo Playground / MJX env stepping on JAX ROCm
PPO rollout/update on JAX ROCm
No Brax pmap
No PyTorch/JAX DLPack interop
```

## Why this fixes the blocker

The upstream MuJoCo Playground `learning/train_jax_ppo.py` routes through Brax PPO, which wraps training epochs in `jax.pmap`. On this machine that path segfaulted even for small train smokes. RSL-RL also failed because it mixes JAX ROCm env tensors with PyTorch ROCm via DLPack/HIP runtime boundaries.

The new runner avoids both problem areas:

- uses only JAX for env stepping, policy, value, GAE, and PPO updates;
- uses only `jax.jit`, not `jax.pmap`;
- uses MuJoCo Playground/MJX directly through `wrapper.wrap_for_brax_training`;
- writes JSONL metrics, checkpoints, and rollout videos.

## Verification

### Cartpole sanity smoke

Command shape:

```bash
python scripts/brotisserie_mjx_single_device_ppo.py \
  --env-name CartpoleBalance \
  --total-timesteps 8192 \
  --num-envs 64 \
  --num-steps 32 \
  --update-epochs 1 \
  --num-minibatches 4
```

Result:

```text
exit_code: 0
checkpoint: step_8192.npz
video: step_8192.mp4
```

### G1 rough-terrain robot smoke

Command shape:

```bash
python scripts/brotisserie_mjx_single_device_ppo.py \
  --env-name G1JoystickRoughTerrain \
  --total-timesteps 32768 \
  --num-envs 128 \
  --num-steps 32 \
  --update-epochs 1 \
  --num-minibatches 4
```

Result:

```text
exit_code: 0
checkpoint: step_32768.npz
video: step_32768.mp4
```

### 1M-step G1 verification run

Command:

```bash
cd /home/marks/c/git/mujoco_playground
source /home/marks/c/git/stable-baselines3/.venv-mjx-rocm010/bin/activate
export HIP_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

python /home/marks/c/git/stable-baselines3/scripts/brotisserie_mjx_single_device_ppo.py \
  --env-name G1JoystickRoughTerrain \
  --run-dir /tmp/brotisserie_mjx_g1_1m_verify \
  --total-timesteps 1048576 \
  --num-envs 1024 \
  --num-steps 64 \
  --update-epochs 2 \
  --num-minibatches 8 \
  --video-every-updates 8 \
  --checkpoint-every-updates 8 \
  --video-max-steps 60
```

Result:

```text
exit_code: 0
timesteps: 1,048,576
final update: 16
final wall_time_s: 238.91
steady SPS after compile: ~7,590 env steps/s
checkpoints: step_524288.npz, step_1048576.npz
videos: step_524288.mp4, step_1048576.mp4
```

Observed during the 1M run:

```text
GPU[0] use: 100%
VRAM: ~13-21%
JAX device: RocmDevice(id=0)
```

Final logged row:

```json
{
  "update": 16,
  "timesteps": 1048576,
  "wall_time_s": 238.90981221199036,
  "sps": 7591.3773841126185,
  "mean_reward": -0.14046931266784668,
  "approx_kl": 0.014368494972586632,
  "clip_frac": 0.21776580810546875,
  "entropy": 41.573020935058594,
  "loss": -0.1335134655237198,
  "policy_loss": -0.005046206060796976,
  "value_loss": 0.15879569947719574
}
```

Video manifest rows:

```text
videos/step_524288.mp4  episode_reward=-5.411690488574095  episode_steps=59
videos/step_1048576.mp4 episode_reward=-4.071770115544496  episode_steps=48
```

## Current caveats

This is a functional GPU-native PPO baseline, but it is intentionally compact and not yet equivalent to the tuned MuJoCo Playground/Brax PPO implementation.

Known gaps to improve next:

1. The value network currently uses the policy observation (`state`) instead of privileged critic observations (`privileged_state`) for G1.
2. Gaussian actions are clipped to `[-1, 1]`; the log-prob does not include a tanh-squash correction.
3. Checkpoints are raw NumPy parameter leaves plus a treedef repr; load/resume support still needs to be implemented.
4. Eval currently relies on rollout video reward; add batched deterministic eval metrics for better reporting.
5. For a 24h run, choose `video_every_updates` from measured update duration so videos land roughly hourly.

## 24h sizing estimate

At the verified steady-state rate:

```text
~7,590 env steps/s
~27.3M env steps/hour
~655M env steps/day
```

A reasonable 24h launch target is therefore about:

```text
--total-timesteps 650000000
```

With the verified config (`num_envs=1024`, `num_steps=64`), one update is 65,536 env steps and takes about 8.6s after compile. Hourly video/checkpoint cadence is roughly:

```text
3600 / 8.6 ~= 418 updates
```

So use:

```text
--video-every-updates 400
--checkpoint-every-updates 400
```

## Launch command for the real GPU-native 24h robot run

```bash
cd /home/marks/c/git/mujoco_playground
source /home/marks/c/git/stable-baselines3/.venv-mjx-rocm010/bin/activate
export HIP_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

RUN_DIR=/home/marks/c/git/stable-baselines3/reports/brotisserie_mjx_24h/g1_rough_seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

python -u /home/marks/c/git/stable-baselines3/scripts/brotisserie_mjx_single_device_ppo.py \
  --env-name G1JoystickRoughTerrain \
  --run-dir "$RUN_DIR" \
  --total-timesteps 650000000 \
  --num-envs 1024 \
  --num-steps 64 \
  --update-epochs 2 \
  --num-minibatches 8 \
  --video-every-updates 400 \
  --checkpoint-every-updates 400 \
  --video-max-steps 600 \
  --seed 0 \
  2>&1 | tee "$RUN_DIR/train.log"
```
