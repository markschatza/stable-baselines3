# Brotisserie MJX G1 Rough-Terrain 1B PPO Run

## Status

Started a 1B-step GPU-native MuJoCo Playground / MJX PPO run on the AMD RX 7900 GRE.

This uses the verified single-device JAX PPO runner, not the CPU-bound SB3/Gymnasium MuJoCo fallback.

## Process handles

Training:

```text
session_id: proc_96aae2eb3a49
launcher pid: 129277
JAX/GPU pid observed by rocm-smi: 129296
```

System monitor:

```text
session_id: proc_cb4bdf2b5060
pid: 129420
```

Run directory:

```text
/home/marks/c/git/stable-baselines3/reports/brotisserie_mjx_1b/g1_rough_seed0_20260708_143227
```

Latest-run pointer:

```text
/home/marks/c/git/stable-baselines3/reports/brotisserie_mjx_1b/latest_run_dir.txt
```

## Command

```bash
cd /home/marks/c/git/mujoco_playground
source /home/marks/c/git/stable-baselines3/.venv-mjx-rocm010/bin/activate
export HIP_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

python -u /home/marks/c/git/stable-baselines3/scripts/brotisserie_mjx_single_device_ppo.py \
  --env-name G1JoystickRoughTerrain \
  --run-dir "$RUN_DIR" \
  --total-timesteps 1000000000 \
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

## Video/checkpoint cadence

Each update is:

```text
1024 envs * 64 steps = 65,536 env steps/update
```

Videos/checkpoints every 400 updates:

```text
26,214,400 env steps per video/checkpoint
```

Based on the 1M verification run steady-state rate of ~7,590 env steps/s, this should be close to hourly:

```text
26,214,400 / 7,590 ~= 3,454 s ~= 57.6 min
```

Expected 1B duration at that rate:

```text
1,000,000,000 / 7,590 ~= 131,752 s ~= 36.6 h
```

Expected number of videos/checkpoints:

```text
1,000,000,000 / 26,214,400 ~= 38
```

## Early verification

The run started cleanly and is using the GPU.

ROCm during startup:

```text
GPU[0] use: 100%
VRAM: 13-16%
GPU PID: 129296
```

Early metrics:

```json
{"update": 1, "timesteps": 65536, "wall_time_s": 44.83536911010742, "sps": 1461.7034939855798, "mean_reward": -0.13016287982463837, "approx_kl": 0.031339891254901886, "clip_frac": 0.38951873779296875, "entropy": 41.161075592041016, "loss": 0.34815698862075806, "policy_loss": 0.005749761592596769, "value_loss": 1.0964252948760986}
{"update": 2, "timesteps": 131072, "wall_time_s": 70.21405982971191, "sps": 2582.443181063603, "mean_reward": -0.15474680066108704, "approx_kl": 0.01548541709780693, "clip_frac": 0.23415374755859375, "entropy": 41.190818786621094, "loss": 0.33490002155303955, "policy_loss": -0.0022345276083797216, "value_loss": 1.0861773490905762}
{"update": 3, "timesteps": 196608, "wall_time_s": 78.88905501365662, "sps": 7555.4773544501295, "mean_reward": -0.15343157947063446, "approx_kl": 0.015667296946048737, "clip_frac": 0.232879638671875, "entropy": 41.21904373168945, "loss": 0.24584747850894928, "policy_loss": -0.003020409494638443, "value_loss": 0.909926176071167}
```

## Artifacts

Inside the run directory:

```text
train.log              stdout/stderr tee
config.json            runner config
events.jsonl           PPO metrics per update
system_monitor.jsonl   minute-level process/GPU monitor
launch_command.sh      exact launch shape
checkpoints/           hourly-ish raw JAX parameter checkpoints
videos/                hourly-ish rollout videos
video_manifest.jsonl   video reward/step metadata
```

## Notes

- First two updates include compilation/warmup overhead; steady-state begins around update 3.
- The runner emits a JAX overflow-cast warning during MuJoCo Playground config handling. This appeared in the successful 1M verification too and did not block training.
- Checkpoint save format is currently raw NumPy parameter leaves plus a treedef repr; load/resume still needs implementation.
