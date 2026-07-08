# Brotisserie MuJoCo Run Path Determination

## Result

We determined a working MuJoCo path on this machine and started a 24h baseline run.

The **working path** is:

```text
Classic Gymnasium/MuJoCo Humanoid-v5
SB3 PPO
CPU-side MuJoCo simulation
PyTorch ROCm policy/value updates via SB3 device=cuda
```

This is not as GPU-bound as MuJoCo Playground/MJX, but it is currently the launch-safe MuJoCo route on the AMD RX 7900 GRE host.

The **not-yet-working path** is:

```text
MuJoCo Playground / MJX / Brax PPO on JAX ROCm
```

Short PPO train smoke tests still segfault in the Brax/JAX training path. Env reset/step/render works, but PPO optimization does not complete reliably.

## Why Humanoid-v5

`Humanoid-v5` is the hardest classic Gymnasium MuJoCo locomotion task in the initial local stack and gives visible robot improvement/failure modes in videos.

Caveat: SB3 warns that MLP PPO is primarily CPU-oriented even when `device=cuda`; GPU utilization is modest because simulation is CPU-side and the network is small.

## 24h run launched

Run directory:

```text
/home/marks/c/git/stable-baselines3/reports/brotisserie_mujoco_24h/humanoid_v5_seed0_20260708_132750
```

Hermes process:

```text
training session_id: proc_98d6f7f7315c
training pid: 112724
monitor session_id: proc_3c49b713aedb
monitor pid: 112896
observed ROCm GPU PID: 112742
```

Command shape:

```bash
cd /home/marks/c/git/stable-baselines3
source .venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python -u scripts/brotisserie_mujoco_ppo_24h.py \
  --env Humanoid-v5 \
  --run-dir /home/marks/c/git/stable-baselines3/reports/brotisserie_mujoco_24h/humanoid_v5_seed0_20260708_132750 \
  --hours 24 \
  --video-interval-s 3600 \
  --initial-chunk-steps 200000 \
  --n-envs 8 \
  --n-steps 2048 \
  --batch-size 1024 \
  --eval-episodes 5 \
  --video-max-steps 1000 \
  --device cuda \
  --seed 0 \
  2>&1 | tee train.log
```

## Verified first checkpoint/video

First chunk completed:

```json
{
  "num_timesteps": 212992,
  "chunk_elapsed_s": 72.60388684272766,
  "chunk_sps": 2933.6170453432724,
  "eval_mean_reward": 323.603456,
  "eval_std_reward": 21.541045276546782
}
```

First checkpoint:

```text
checkpoints/step_212992.zip
```

First rollout video:

```text
videos/step_212992.mp4
```

The runner adapts future chunk sizes toward `video_interval_s`, so after the initial calibration chunk it should target roughly one checkpoint/video per hour.

## Artifacts produced by the runner

Inside the run directory:

```text
metadata.json
train.log
events.jsonl
metrics.jsonl
video_manifest.jsonl
system_monitor.jsonl
monitor/*.monitor.csv
tb/PPO_*/events.out.tfevents.*
checkpoints/step_<timesteps>.zip
videos/step_<timesteps>.mp4
final_model.zip  # when complete
```

## Scripts

Created:

```text
scripts/brotisserie_mujoco_ppo_24h.py
scripts/monitor_mujoco_24h.sh
```

Also created a local debugging helper for the still-blocked Brax/JAX path:

```text
scripts/patch_brax_ppo_single_device_rocm.py
```

## Current recommendation

Let the Humanoid-v5 24h run continue as the launch-safe MuJoCo baseline on this machine.

In parallel, the better research path is still to build or find a stable non-`pmap` MuJoCo Playground/MJX PPO loop for AMD ROCm, because that would be truly GPU-bound robotics.
