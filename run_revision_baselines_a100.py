from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CAERNET_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_DIR.parents[1]

CONFIGS = [
    CAERNET_DIR / "configs/baselines/resnet50_paper.yaml",
    CAERNET_DIR / "configs/caernet_resnet50.yaml",
    CAERNET_DIR / "configs/ms_caernet_resnet50.yaml",
    CAERNET_DIR / "configs/baselines/efficientnet_v2_s.yaml",
    CAERNET_DIR / "configs/baselines/convnext_tiny.yaml",
    CAERNET_DIR / "configs/revision/adaptation_linear_probe.yaml",
    CAERNET_DIR / "configs/revision/clip_adapter_baseline.yaml",
]

DATA_DIRS = [
    Path("classify/data/artbench10/train"),
    Path("classify/data/artbench10_paper/val"),
    Path("classify/data/artbench10/test"),
]


def _check_inputs() -> None:
    for data_dir in DATA_DIRS:
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset path not found: {data_dir}\n"
                "Run from CAERNet once: python prepare_data.py"
            )
    for config in CONFIGS:
        if not config.is_file():
            raise FileNotFoundError(f"Experiment config not found: {config}")


def _run_config(
    config: Path,
    seed: int,
    hardware: str,
    dry_run: bool,
    extra_args: list[str],
) -> None:
    command = [
        os.environ.get("PYTHON_BIN", sys.executable),
        str(CAERNET_DIR / "run_training.py"),
        "--config",
        str(config),
        "--hardware",
        hardware,
        "--seed",
        str(seed),
        "--run-suffix",
        f"seed{seed}",
        "--final-test-only",
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CAERNET_DIR)
    print()
    print("=" * 72)
    print(f"Config: {config.name}")
    print(f"Seed:   {seed}")
    print("=" * 72)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun trainable main-table baselines over three seeds on A100.",
    )
    parser.add_argument("--hardware", default="a100")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()

    os.chdir(REPO_ROOT)
    _check_inputs()
    print(f"Starting baseline reruns on hardware={args.hardware}")
    print(f"Seeds: {args.seeds}")

    for seed in args.seeds:
        for config in CONFIGS:
            _run_config(config, seed, args.hardware, args.dry_run, extra_args)

    print()
    print("All baseline reruns finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
