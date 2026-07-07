# Craftax 1B PPO Run Log

## Run launched

A blank/default Craftax PPO 1B-step run was launched on the local AMD Radeon RX 7900 GRE.

Hermes background process:

```text
session_id: proc_2f0fcc5e0da7
pid: 47344
child GPU pid observed by rocm-smi: 47361
```

Command:

```bash
cd /home/marks/c/git/Craftax_Baselines
mkdir -p brotisserie_runs
source /home/marks/c/git/stable-baselines3/.venv-jax-rocm071/bin/activate
export WANDB_MODE=offline HIP_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
python -u ppo.py \
  --total_timesteps 1000000000 \
  --num_envs 1024 \
  --num_steps 64 \
  --seed 42 \
  --wandb_project brotisserie-craftax-1b \
  --wandb_entity brotisserie \
  2>&1 | tee brotisserie_runs/craftax_ppo_1b_seed42_20260707_183926.log
```

W&B offline run:

```text
/home/marks/c/git/Craftax_Baselines/wandb/offline-run-20260707_183927-je0l12ab
```

Log file:

```text
/home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_20260707_183926.log
```

## Environment

The successful run uses a JAX ROCm 0.7.1 environment because JAX ROCm 0.10.2 segfaulted when compiling/running the Craftax PPO training function.

```bash
cd /home/marks/c/git/stable-baselines3
uv venv --python python3.12 .venv-jax-rocm071
. .venv-jax-rocm071/bin/activate
uv pip install pip
uv pip install 'jax==0.7.1' 'jaxlib==0.7.1' 'jax-rocm7-pjrt==0.7.1' 'jax-rocm7-plugin==0.7.1'
uv pip install craftax distrax optax flax wandb gymnax chex
# Re-pin JAX afterwards because current Craftax deps resolve JAX upward:
uv pip install --reinstall 'jax==0.7.1' 'jaxlib==0.7.1' 'jax-rocm7-pjrt==0.7.1' 'jax-rocm7-plugin==0.7.1'
```

Verified:

```text
jax 0.7.1
default_backend gpu
devices [RocmDevice(id=0)]
```

## Craftax_Baselines local patch

The upstream `ppo.py` always wraps the jitted train function in `jax.vmap`, even for `NUM_REPEATS=1`. On this local ROCm stack that path segfaulted during compilation/execution.

Local patch applied in `/home/marks/c/git/Craftax_Baselines/ppo.py`:

```text
references/craftax_baselines_num_repeats_1_rocm.patch
```

This bypasses `vmap` when `NUM_REPEATS == 1` and calls `train_jit(rngs[0])` directly. The 1M-step smoke test succeeded with that patch.

## Smoke tests before launch

### Failed path

JAX ROCm 0.10.2:

- Craftax reset/step worked.
- Full PPO training segfaulted inside JAX `pjit`/ROCm path.

### Successful path

JAX ROCm 0.7.1 + local `NUM_REPEATS == 1` patch:

```bash
WANDB_MODE=offline HIP_VISIBLE_DEVICES=0 python -u ppo.py \
  --total_timesteps 8192 \
  --num_envs 32 \
  --num_steps 16 \
  --num_minibatches 4 \
  --update_epochs 1 \
  --seed 0 \
  --wandb_project brotisserie-craftax-smoke \
  --wandb_entity brotisserie
```

Observed:

```text
Started logging
Time to run experiment 60.307305574417114
SPS: 135.83760577549526
```

Default-shape 1M smoke:

```bash
WANDB_MODE=offline HIP_VISIBLE_DEVICES=0 python -u ppo.py \
  --total_timesteps 1048576 \
  --num_envs 1024 \
  --num_steps 64 \
  --seed 1 \
  --wandb_project brotisserie-craftax-smoke \
  --wandb_entity brotisserie
```

Observed:

```text
Started logging
Time to run experiment 94.48765707015991
SPS: 11097.491804896812
```

At ~11.1k SPS, a 1B-step run estimates to ~25 hours, which matches the desired 24h-class workload.

## Initial 1B run hardware status

Immediately after launch, `rocm-smi` showed:

```text
GPU[0] Radeon RX 7900 GRE: GPU use 100%
GPU[0] VRAM allocated 89%
```

So the run is actually occupying the card.

## Public target

Craftax README public reference for Craftax-1B blank PPO:

```text
PPO score: 11.9
```

Goal for this first run: see whether the local blank/default PPO setup can match or approach that score.

## Monitoring

Check process:

```bash
# Hermes
process poll proc_2f0fcc5e0da7

# Shell/GPU
rocm-smi --showuse --showmemuse --showpidgpus
```

Tail logs:

```bash
tail -f /home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_20260707_183926.log
```

Sync offline W&B later if desired:

```bash
cd /home/marks/c/git/Craftax_Baselines
source /home/marks/c/git/stable-baselines3/.venv-jax-rocm071/bin/activate
wandb sync wandb/offline-run-20260707_183927-je0l12ab
```
