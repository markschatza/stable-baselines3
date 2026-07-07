"""Extract public PPO defaults/benchmarks into easy-to-scan Brotisserie docs.

The script reads a local clone of DLR-RM/rl-baselines3-zoo and this SB3 fork,
then writes selected references under references/public_baselines/ plus a
summary markdown file at BROTISSERIE_PUBLIC_BASELINES.md.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
from pathlib import Path

from stable_baselines3 import PPO

SELECTED_ENVS = [
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    "LunarLander-v2",
    "LunarLander-v3",
    "BipedalWalker-v3",
    "HalfCheetah-v3",
    "HalfCheetah-v5",
    "Hopper-v3",
    "Hopper-v5",
    "Walker2d-v3",
    "Walker2d-v5",
    "Ant-v3",
    "Ant-v5",
    "PongNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "QbertNoFrameskip-v4",
]


def run(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def extract_yaml_block(text: str, key: str) -> str | None:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line == f"{key}:":
            start = idx
            break
    if start is None:
        return None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line and not line.startswith(" ") and not line.startswith("#"):
            end = idx
            break
    # Include immediately preceding comment lines such as "# Tuned".
    while start > 0 and lines[start - 1].startswith("#"):
        start -= 1
    block_lines = lines[start:end]
    while block_lines and (not block_lines[-1] or block_lines[-1].startswith("#")):
        block_lines.pop()
    return "\n".join(block_lines).rstrip()


def ppo_defaults_table() -> str:
    signature = inspect.signature(PPO.__init__)
    rows = ["| parameter | SB3 default |", "|---|---:|"]
    skip = {
        "self",
        "policy",
        "env",
        "rollout_buffer_class",
        "rollout_buffer_kwargs",
        "tensorboard_log",
        "verbose",
        "seed",
        "device",
        "_init_setup_model",
    }
    for name, parameter in signature.parameters.items():
        if name in skip or parameter.default is inspect._empty:
            continue
        rows.append(f"| `{name}` | `{parameter.default!r}` |")
    return "\n".join(rows)


def selected_benchmark_rows(benchmark_text: str) -> list[str]:
    rows = []
    for line in benchmark_text.splitlines():
        if not line.startswith("|ppo"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        env_id = cells[1]
        if env_id in SELECTED_ENVS:
            rows.append(line)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-zoo", type=Path, default=Path("../rl-baselines3-zoo"), help="Path to rl-baselines3-zoo clone.")
    parser.add_argument("--out-dir", type=Path, default=Path("references/public_baselines"), help="Output reference dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    rl_zoo = args.rl_zoo.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_yml = rl_zoo / "hyperparams" / "ppo.yml"
    benchmark_md = rl_zoo / "benchmark.md"
    if not ppo_yml.exists() or not benchmark_md.exists():
        raise FileNotFoundError(f"Expected rl-baselines3-zoo files under {rl_zoo}")

    ppo_text = ppo_yml.read_text(encoding="utf-8")
    benchmark_text = benchmark_md.read_text(encoding="utf-8")
    shutil.copyfile(ppo_yml, out_dir / "rl_zoo_ppo.yml")

    selected_blocks: list[str] = []
    for key in ["default", "atari", *SELECTED_ENVS]:
        block = extract_yaml_block(ppo_text, key)
        if block is not None:
            block = block.replace("# Tuned\n\n# Tuned\n", "# Tuned\n")
            selected_blocks.append(block)
    (out_dir / "rl_zoo_ppo_selected.yml").write_text("\n\n".join(selected_blocks) + "\n", encoding="utf-8")

    rows = selected_benchmark_rows(benchmark_text)
    benchmark_selected = "\n".join(
        [
            "# Selected RL Zoo PPO benchmark rows",
            "",
            "Source: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md",
            "",
            "| algo | env_id | mean_reward | std_reward | n_timesteps | eval_timesteps | eval_episodes |",
            "|---|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (out_dir / "rl_zoo_ppo_benchmark_selected.md").write_text(benchmark_selected, encoding="utf-8")

    sb3_commit = run(["git", "rev-parse", "HEAD"], repo_root)
    rl_zoo_commit = run(["git", "rev-parse", "HEAD"], rl_zoo)
    selected_entry_lines = "\n".join(
        f"- `{env}`"
        for env in [
            "default",
            "atari",
            *[env for env in SELECTED_ENVS if extract_yaml_block(ppo_text, env) is not None],
        ]
    )
    summary = f"""# Public PPO Defaults and Benchmarks

This file makes the public baseline material visible inside the Brotisserie SB3 fork.

## Sources grabbed

| Source | What we use | Local copy |
|---|---|---|
| Stable-Baselines3 | Upstream PPO constructor defaults | live introspection from this checkout |
| RL Baselines3 Zoo | Tuned PPO hyperparameters | `references/public_baselines/rl_zoo_ppo.yml` |
| RL Baselines3 Zoo | Selected tuned PPO config blocks | `references/public_baselines/rl_zoo_ppo_selected.yml` |
| RL Baselines3 Zoo | Public trained-agent benchmark rows | `references/public_baselines/rl_zoo_ppo_benchmark_selected.md` |
| Hugging Face SB3 | Model cards, videos, trained agents | https://huggingface.co/sb3 |
| OpenRL Benchmark / W&B | SB3 benchmark dashboard link from SB3 README | https://wandb.ai/openrlbenchmark/sb3 |

Source revisions at extraction time:

- This SB3 fork commit: `{sb3_commit}`
- Local `rl-baselines3-zoo` commit: `{rl_zoo_commit}`

Important caveat from RL Zoo: its `benchmark.md` table is not a rigorous multi-seed benchmark. It is a one-run
trained-agent performance table intended to check maximal algorithm performance, find bugs, and make pretrained agents
available.

## SB3 PPO constructor defaults

These are the upstream SB3 defaults from `stable_baselines3.PPO.__init__` in this checkout.

{ppo_defaults_table()}

## RL Zoo tuned config coverage we care about first

The selected local YAML includes `default`, `atari`, and any matching entries for our candidate benchmark environments.

```text
references/public_baselines/rl_zoo_ppo_selected.yml
```

Key available selected entries currently include:

{selected_entry_lines}

## Selected RL Zoo PPO benchmark rows

See:

```text
references/public_baselines/rl_zoo_ppo_benchmark_selected.md
```

Those rows give us public tuned-agent target numbers for common envs such as CartPole, Pendulum, BipedalWalker,
classic MuJoCo v3 tasks, and Atari.

## How this fits Brotisserie

We should expose three columns in future result tables:

1. `sb3_default`: our local run of upstream SB3 PPO defaults.
2. `rl_zoo_tuned`: local run using RL Zoo PPO tuned hyperparameters where available.
3. `rl_zoo_public_target`: public RL Zoo trained-agent score from `benchmark.md`, clearly labeled as a one-run public
   target, not a statistically rigorous benchmark.

This gives us both local reproducibility and a visible public target for sanity checking.

## Refresh command

```bash
cd /home/marks/c/git/stable-baselines3
. .venv/bin/activate
python scripts/brotisserie_public_baselines.py --rl-zoo ../rl-baselines3-zoo
```
"""
    (repo_root / "BROTISSERIE_PUBLIC_BASELINES.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
