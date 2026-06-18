from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]

if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from run_training import (  # noqa: E402
    _ensure_dataset_ready,
    _resolve_hardware_path,
    build_effective_config,
)
from src.train import train_from_config  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/experiments/clip_adapter_vit_b16_no_energy.yaml",
    "configs/experiments/clip_adapter_vit_b16_no_contrastive.yaml",
    "configs/experiments/clip_adapter_vit_b16.yaml",
]
DEFAULT_SEEDS = [41, 42, 43]


def _resolve_config(path: str) -> Path:
    config_path = Path(path)
    if config_path.exists():
        return config_path.resolve()
    candidate = CAERNET_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Missing config file: {path}")


def _seeded_run_name(config: dict, seed: int) -> str:
    output = config.setdefault("output", {})
    base_name = output.get("run_name")
    if not base_name:
        run_dir = output.get("run_dir")
        base_name = Path(run_dir).name if run_dir else "clip_seed_run"
    return f"{base_name}_seed{seed}"


def _set_seeded_output(config: dict, seed: int) -> Path:
    config["train"]["seed"] = seed

    output = config.setdefault("output", {})
    run_name = _seeded_run_name(config, seed)
    output["run_name"] = run_name

    run_root = output.get("run_root")
    if run_root:
        run_dir = Path(run_root) / run_name
    else:
        parent = Path(output.get("run_dir", "classify/outputs/runs")).parent
        run_dir = parent / run_name

    output["run_dir"] = str(run_dir)
    return run_dir


def _prepare_run_dir(run_dir: Path, force: bool) -> bool:
    if not run_dir.exists():
        return True

    if force:
        print(f"Removing existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
        return True

    history = run_dir / "history.csv"
    best = run_dir / "best.pt"
    if history.exists() or best.exists():
        print(f"Skipping existing run: {run_dir}")
        print("  Use --force to rerun and overwrite it.")
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CLIP adapter ablations with multiple seeds on A100.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to run. Default: 41 42 43",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Experiment config paths. Defaults to the three adapter ablations.",
    )
    parser.add_argument(
        "--hardware",
        default="a100",
        help="Hardware profile name or YAML path. Default: a100",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data/model and run one forward pass for each planned run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing seed run directories before rerunning.",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    hardware_path = _resolve_hardware_path(args.hardware)

    print(f"Working directory: {REPO_ROOT}")
    print(f"Hardware profile: {hardware_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Configs: {args.configs}")

    for seed in args.seeds:
        for config_arg in args.configs:
            config_path = _resolve_config(config_arg)
            config = build_effective_config(config_path, hardware_path)
            run_dir = _set_seeded_output(config, seed)

            print()
            print("=" * 72)
            print(f"Config: {config_path}")
            print(f"Seed:   {seed}")
            print(f"Run:    {run_dir}")
            print("=" * 72)

            _ensure_dataset_ready(config)
            if not args.dry_run and not _prepare_run_dir(run_dir, force=args.force):
                continue

            train_from_config(config, dry_run=args.dry_run)

    print()
    print("All requested seed runs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
