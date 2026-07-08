# MuJoCo Playground G1 PPO ROCm Blocker

User asked to fix and start a 24h baseline PPO run for `G1JoystickRoughTerrain` with videos.

I did **not** launch the 24h run because the available AMD/JAX/ROCm MuJoCo Playground training paths fail short smoke tests. Launching a 24h job would just create a long-running crash/no-result.

## Target

```text
MuJoCo Playground / MJX
Env: G1JoystickRoughTerrain
Goal: baseline PPO, 24h-class, hourly rollout videos/checkpoints
GPU: AMD Radeon RX 7900 GRE
```

## What works

### JAX ROCm GPU

```text
jax 0.10.2
backend gpu
[RocmDevice(id=0)]
```

### MuJoCo Playground imports/env stepping/rendering

A JAX/MJX env reset/step smoke worked for:

```text
CartpoleBalance
PandaPickCube
G1JoystickFlatTerrain
```

A random G1 rough-terrain video render succeeded:

```text
/home/marks/c/git/mujoco_playground/brotisserie_runs/videos/random_g1_rough_smoke.mp4
```

Installed ffmpeg support via `imageio-ffmpeg` and linked it into the venv.

## What fails

### 1. Brax/JAX PPO path segfaults in `jax.pmap`

Command shape:

```bash
cd /home/marks/c/git/mujoco_playground
source /home/marks/c/git/stable-baselines3/.venv-mjx-rocm010/bin/activate
HIP_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline \
python -u learning/train_jax_ppo.py \
  --env_name=G1JoystickRoughTerrain \
  --impl=jax \
  --num_timesteps=262144 \
  --num_evals=2 \
  --num_envs=1024 \
  --num_eval_envs=128 \
  --num_videos=1
```

Failure:

```text
Fatal Python error: Segmentation fault
...
jax/_src/pmap.py", line 319 in wrapped
brax/training/agents/ppo/train.py", line 684 in training_epoch_with_timing
```

This is not G1-specific. Similar short training smoke tests also failed for:

```text
CartpoleBalance
PandaPickCube
```

So the blocker is the current Brax PPO `pmap` training path on this JAX/ROCm stack, not the specific environment.

### 2. JAX 0.7.1 + MuJoCo MJX path is also unstable

Tried a lower-version stack:

```text
jax 0.7.1
mujoco 3.6.0
mujoco-mjx 3.6.0
brax 0.13.0
```

The env setup/import proceeded, but G1/Panda-style compile/step did not complete reliably within the smoke-test window and prior JAX 0.7.1 + MJX 3.10 attempts hit:

```text
ROCM_ERROR_ILLEGAL_ADDRESS
```

### 3. RSL-RL/PyTorch route conflicts with JAX ROCm

Tried MuJoCo Playground's `learning/train_rsl_rl.py` with ROCm PyTorch.

First, installing `rsl-rl-lib warp-lang` pulled CUDA PyTorch, causing:

```text
RuntimeError: Found no NVIDIA driver on your system.
```

After reinstalling ROCm PyTorch:

```text
torch 2.9.1+rocm6.4
HIP 6.4.43484
Radeon RX 7900 GRE
```

RSL-RL still failed at JAX→Torch DLPack handoff / HIP runtime registration:

```text
hipApiName has non-null function pointer ...
Fatal Python error: Aborted
```

Trying to initialize PyTorch/HIP before JAX caused the reverse failure:

```text
python: symbol lookup error: /opt/rocm/lib/libamdhip64.so.7: undefined symbol: hsa_amd_memory_get_preferred_copy_engine, version ROCR_1
```

So the mixed JAX ROCm + PyTorch ROCm route is not launch-safe either.

## Attempted local patches

Captured attempted patches here:

```text
references/mujoco_playground_rocm_attempted_patches.patch
```

They included:

- disabling the CUDA-oriented `--xla_gpu_triton_gemm_any=True` flag unless explicitly requested,
- adding a JAX >= 0.10 shim for removed `jax.device_put_replicated`,
- skipping loss-metric reductions that triggered secondary segfaults.

These did not make Brax/JAX PPO stable; the root pmap call still segfaulted.

## Decision

Do **not** start the requested 24h G1 PPO run on this machine yet. The short verified smoke test must complete first; it currently does not.

## Best next options

1. Try a known-good MuJoCo Playground/Brax/JAX/ROCm version matrix, ideally one reported by ROCm/JAX users for pmap-heavy training.
2. Implement a single-device non-pmap PPO training loop for MuJoCo Playground/MJX, analogous to the Craftax `NUM_REPEATS == 1` workaround. This is more work but likely the cleanest AMD path.
3. Use a CUDA/NVIDIA host for MuJoCo Playground/Isaac Lab robot sims.
4. If the goal is immediate 24h GPU-bound training on this AMD box, run the next Craftax baseline variant with hourly videos/checkpoints while the MJX robot path is fixed.

## Video logging plan once PPO training is stable

Use MuJoCo Playground/Brax `policy_params_fn(current_step, make_policy, params)` to render a deterministic rollout at eval/checkpoint boundaries.

For roughly hourly videos in a 24h run:

```text
num_evals = 25
```

Artifacts:

```text
<logdir>/videos/step_<step>.mp4
<logdir>/video_manifest.jsonl
<logdir>/checkpoints/<step>/
```
