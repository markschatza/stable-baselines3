# Craftax 1B PPO Result

Run completed normally.

## Command/run identity

Hermes background process:

```text
session_id: proc_2f0fcc5e0da7
exit_code: 0
```

Craftax run files:

```text
/home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_20260707_183926.log
/home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_history.csv
/home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_monitor_20260707_213957.jsonl
/home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_summary.json
/home/marks/c/git/Craftax_Baselines/wandb/offline-run-20260707_183927-je0l12ab
```

## Runtime / throughput

Terminal output:

```text
Time to run experiment 24950.016494512558
SPS:  40080.13382355629
```

Equivalent:

```text
wall clock: ~6.93 hours
steps/sec: ~40.1k
```

This was much faster than the initial 1M-step estimate because the initial smoke included compile/startup overhead and only ran a tiny number of updates.

## GPU utilization

The external monitor captured 237 one-minute samples over the latter part of the run.

```text
GPU utilization: 100% average, min 100%, max 100%
VRAM allocation: ~88.0% average, min 88%, max 89%
```

So this was a genuinely GPU-bound run on the RX 7900 GRE.

## W&B metrics extracted locally

Extracted from:

```text
wandb/offline-run-20260707_183927-je0l12ab/run-je0l12ab.wandb
```

Generated CSV:

```text
/home/marks/c/git/Craftax_Baselines/brotisserie_runs/craftax_ppo_1b_seed42_history.csv
```

History rows:

```text
15258
```

Final logged metrics:

| metric | value |
|---|---:|
| `_step` | 15257 |
| `_runtime` | 24953.184788791 |
| `sps` | 40220.881003785515 |
| `achievements` | 22.59999656677246 |
| `episode_return` | 22.980003356933594 |
| `episode_length` | 2480.280029296875 |

Best logged achievement metric:

| metric | value |
|---|---:|
| `_step` | 14828 |
| `_runtime` | 24253.618012786 |
| `sps` | 40206.61438713846 |
| `achievements` | 24.375 |
| `episode_return` | 25.35000228881836 |
| `episode_length` | 3687.9375 |

## Public target comparison

Craftax README public reference for Craftax-1B blank PPO:

```text
PPO score: 11.9
```

Our run:

```text
final achievements: 22.60
best achievements: 24.375
final episode_return: 22.98
best episode_return: 25.35
```

Interpretation: this run appears to exceed the public blank PPO score substantially, but we should verify exact score semantics/version differences before claiming a strict apples-to-apples benchmark win. The run used Craftax `1.6.1` and a local ROCm stability patch to avoid `vmap` when `NUM_REPEATS == 1`.

## Final nonzero achievements

Final achievement percentages/count-like logged values:

```text
wake_up: 100
eat_cow: 100
collect_wood: 100
place_table: 100
collect_sapling: 100
place_plant: 100
make_arrow: 96
collect_coal: 96
make_stone_pickaxe: 96
defeat_zombie: 96
make_wood_sword: 96
collect_stone: 96
make_stone_sword: 96
make_torch: 96
place_torch: 96
make_wood_pickaxe: 96
place_stone: 96
place_furnace: 92
defeat_skeleton: 92
collect_drink: 88
collect_iron: 88
make_iron_sword: 84
make_iron_pickaxe: 64
enter_dungeon: 44
eat_plant: 32
make_iron_armour: 20
```

## Artifact caveat

No model checkpoint was saved because the run did not pass `--save_policy`; the upstream script only saves policy state at the end when that flag is enabled. The next long run should add periodic checkpointing and explicit JSONL metrics independent of W&B.
