from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]
WEIGHT_PATH = Path("/home/kmyh/classify/models/open_clip/vit_b16_openai.bin")

if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from run_training import (  # noqa: E402
    _ensure_dataset_ready,
    _resolve_hardware_path,
    build_effective_config,
)
from src.train import train_from_config  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/experiments/clip_adapter_vit_b16_ft_art_mixup_no_energy.yaml",
    "configs/experiments/clip_adapter_vit_b16_ft_art_mixup_no_contrastive.yaml",
    "configs/experiments/clip_adapter_vit_b16_ft_art_mixup.yaml",
]
DEFAULT_SEEDS = [41, 43]


def _resolve_config(path: str) -> Path:
    config_path = Path(path)
    if config_path.exists():
        return config_path.resolve()
    candidate = CAERNET_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Missing config file: {path}")


def _set_seeded_output(config: dict, seed: int) -> Path:
    config["train"]["seed"] = seed

    output = config.setdefault("output", {})
    base_name = output.get("run_name")
    if not base_name:
        base_name = Path(output.get("run_dir", "clip_ft_art_mixup")).name

    run_name = f"{base_name}_seed{seed}"
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

    if (run_dir / "history.csv").exists() or (run_dir / "best.pt").exists():
        print(f"Skipping existing run: {run_dir}")
        print("  Use --force to rerun and overwrite it.")
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strong CLIP art/mixup experiments for seed41 and seed43 on A100.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to run. Default: 41 43",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Experiment config paths. Defaults to the three strong art/mixup configs.",
    )
    parser.add_argument("--hardware", default="a100")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"CLIP weight file not found: {WEIGHT_PATH}")

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
    print("All requested strong seed runs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
