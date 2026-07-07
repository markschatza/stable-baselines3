# Brotisserie Baseline Runs

## Local uv/ROCm environment

Created a local uv environment in the SB3 fork:

```bash
cd /home/marks/c/git/stable-baselines3
uv venv --python python3.11 .venv
. .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/rocm6.4 'torch==2.9.1'
uv pip install -e '.[extra,tests]'
```

Verified hardware/software:

- GPU: Radeon RX 7900 GRE
- ROCm GFX target: `gfx1100`
- PyTorch: `2.9.1+rocm6.4`
- HIP runtime reported by PyTorch: `6.4.43484-123eb5128`
- `torch.cuda.is_available()`: `True`
- visible device for smoke run: `Radeon RX 7900 GRE`

Note: PyTorch ROCm exposes HIP devices through the `torch.cuda` API, so SB3 still sees the device as `cuda`.

## Verification commands run

```bash
# PyTorch ROCm smoke test: GPU matmul succeeded
HIP_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print(float(y[0, 0]))
PY

# SB3 targeted PPO tests
pytest tests/test_run.py::test_ppo -q

# Baseline runner lint/help
ruff check scripts/brotisserie_baseline_ppo.py
python scripts/brotisserie_baseline_ppo.py --help
```

Observed test result:

```text
6 passed in 13.21s
```

## Smoke baseline runner

Script:

```text
scripts/brotisserie_baseline_ppo.py
```

It runs upstream SB3 PPO defaults, evaluates the policy, and writes JSONL rows.

Example GPU smoke command:

```bash
HIP_VISIBLE_DEVICES=0 python scripts/brotisserie_baseline_ppo.py \
  --device cuda \
  --timesteps 10000 \
  --eval-episodes 5 \
  --env CartPole-v1 \
  --env Pendulum-v1 \
  --seed 0 \
  --out reports/brotisserie_baselines/ppo_default_smoke_gpu.jsonl
```

SB3 emits a warning for MLP PPO on GPU because CPU can be faster for small non-CNN policies. This is expected. The run still resolved to `cuda` on the RX 7900 GRE.

## First smoke results

Output file:

```text
reports/brotisserie_baselines/ppo_default_smoke_gpu.jsonl
```

| variant | env | seed | timesteps | device | mean reward | std reward | train seconds | steps/sec |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| SB3 default PPO | CartPole-v1 | 0 | 10000 | cuda / RX 7900 GRE | 500.0 | 0.0 | 19.53 | 512.07 |
| SB3 default PPO | Pendulum-v1 | 0 | 10000 | cuda / RX 7900 GRE | -1028.93 | 226.85 | 8.89 | 1124.97 |

## Immediate follow-ups

- Add CPU-vs-GPU timing comparison for MLP envs so we know when ROCm helps or hurts.
- Add vectorized env runs with larger `n_envs`/`n_steps` to improve device utilization.
- Add CNN/Atari baseline where GPU should matter more.
- Add RL Zoo tuned baseline import path for matched env comparisons.
