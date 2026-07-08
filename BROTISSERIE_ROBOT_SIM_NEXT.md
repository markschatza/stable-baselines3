# Brotisserie Next Robot-Sim Run

## Recommendation

The right next target is **MuJoCo Playground / MJX** rather than classic CPU MuJoCo:

- GPU-accelerated robot simulation and PPO training.
- Hard humanoid/quadruped/manipulation tasks.
- Built-in checkpointing and final rollout video generation.
- Good fit for RX 7900 GRE if the JAX/ROCm stack is stable.

Primary candidate:

```text
G1JoystickRoughTerrain
```

Why this one:

- humanoid locomotion,
- rough terrain,
- high-dimensional privileged critic + policy observation split,
- meaningful failure modes visible in video,
- much more aligned with the user's “hard robot sim, 24h-class GPU run” goal than CPU Gym/MuJoCo tasks.

Secondary candidates:

```text
G1JoystickFlatTerrain
Go1JoystickRoughTerrain
PandaPickCube
PandaOpenCabinet
LeapCubeReorient
```

## Local setup attempted

Created:

```text
/home/marks/c/git/stable-baselines3/.venv-mjx-rocm010
```

Installed:

```bash
uv pip install 'jax==0.10.2' 'jax-rocm7-pjrt==0.10.2' 'jax-rocm7-plugin==0.10.2'
uv pip install 'brax>=0.14.2' etils flax lxml mediapy ml_collections \
  'mujoco-mjx>=3.6.0' 'mujoco>=3.6.0' 'orbax-checkpoint>=0.11.22' \
  tqdm tensorboardX wandb imageio imageio-ffmpeg
cd /home/marks/c/git/mujoco_playground
uv pip install -e . --no-deps
```

Verified:

```text
jax 0.10.2
backend gpu
[RocmDevice(id=0)]
```

Smoke env reset/step succeeded for:

```text
CartpoleBalance
PandaPickCube
G1JoystickFlatTerrain
```

Random G1 rough-terrain video rendering succeeded after adding an ffmpeg binary via `imageio-ffmpeg`:

```text
/home/marks/c/git/mujoco_playground/brotisserie_runs/videos/random_g1_rough_smoke.mp4
```

## Current blocker

Full MuJoCo Playground PPO training on JAX/ROCm is not stable yet.

A short `G1JoystickRoughTerrain` train run:

```bash
python -u learning/train_jax_ppo.py \
  --env_name=G1JoystickRoughTerrain \
  --impl=jax \
  --num_timesteps=262144 \
  --num_evals=2 \
  --num_envs=1024 \
  --num_eval_envs=128 \
  --num_videos=1
```

failed after the initial eval with a ROCm/JAX segfault inside Brax PPO training.

Observed failures:

1. JAX 0.7.1 + MuJoCo MJX 3.10 reset/step on Panda hit:

```text
ROCM_ERROR_ILLEGAL_ADDRESS
```

2. JAX 0.10.2 + Brax 0.14.2 required a local compatibility shim for removed `jax.device_put_replicated`.

3. After that shim, training still segfaulted inside JAX/Brax PPO during the first optimization epoch.

So: **rendering works, env stepping works, but long PPO training is not launch-safe yet**.

## Video logging plan once training is stable

MuJoCo Playground's PPO loop calls `policy_params_fn(current_step, make_policy, params)` at eval/checkpoint boundaries. This is the right hook for hourly videos.

For a 24h run, set roughly:

```text
num_evals = 25
```

That gives:

- initial eval/video at step 0,
- then about one checkpoint/video per hour,
- final video at the end.

The video callback should:

1. render one deterministic rollout using the current policy,
2. save to:

```text
<logdir>/videos/step_<num_steps>.mp4
```

3. append a JSONL row:

```text
<logdir>/video_manifest.jsonl
```

with step, wall-clock, video path, eval reward if available, and checkpoint path.

The built-in script already saves checkpoints at eval boundaries under:

```text
<logdir>/checkpoints/<step>/
```

## Proposed launch command after stability fix

A 24h-class candidate:

```bash
cd /home/marks/c/git/mujoco_playground
source /home/marks/c/git/stable-baselines3/.venv-mjx-rocm010/bin/activate
export HIP_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export WANDB_MODE=offline

python -u learning/train_jax_ppo_brotisserie_video.py \
  --env_name=G1JoystickRoughTerrain \
  --impl=jax \
  --num_timesteps=3500000000 \
  --num_evals=25 \
  --num_envs=1024 \
  --num_eval_envs=128 \
  --num_videos=1 \
  --use_tb=True \
  --log_training_metrics=True \
  --training_metrics_steps=1000000 \
  --logdir=/home/marks/c/git/mujoco_playground/brotisserie_runs/g1_rough_24h \
  --suffix=brotisserie-g1-rough-24h \
  2>&1 | tee /home/marks/c/git/mujoco_playground/brotisserie_runs/g1_rough_24h/train.log
```

The exact timestep count should be calibrated after a stable 5-10M step smoke run reports SPS.

## Next technical work

Before launching a 24h robot run:

1. Try a Brax/MuJoCo Playground version combination that is JAX 0.10-compatible on ROCm without patching.
2. If that fails, try older `mujoco-mjx`/`brax` pins with JAX ROCm 0.7.1.
3. If MuJoCo Playground remains unstable on ROCm, use either:
   - Craftax next with hourly videos/checkpoints, or
   - a separate NVIDIA/CUDA host for Isaac Lab / MuJoCo Playground.

Do **not** launch the 24h robot run until a short PPO train smoke test completes and produces a video.
