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
    CAERNET_DIR / "configs/baselines/convnext_tiny.yaml",
    CAERNET_DIR / "configs/baselines/efficientnet_v2_s.yaml",
    CAERNET_DIR / "configs/ms_caernet_resnet50.yaml",
    CAERNET_DIR / "configs/caernet_resnet50.yaml",
]

DATA_DIRS = [
    Path("classify/data/artbench10/train"),
    Path("classify/data/artbench10_paper/val"),
    Path("classify/data/artbench10/test"),
]


def _check_inputs():
    for data_dir in DATA_DIRS:
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset path not found: {data_dir}\n"
                "Run from CAERNet once: python prepare_data.py"
            )


def _run_config(config: Path, hardware: str, extra_args: list[str]):
    command = [
        os.environ.get("PYTHON_BIN", sys.executable),
        str(CAERNET_DIR / "run_training.py"),
        "--config",
        str(config),
        "--hardware",
        hardware,
        *extra_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CAERNET_DIR)

    print()
    print("=" * 60)
    print(f"Running: {config}")
    print("=" * 60)

    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CNN and CAERNet baseline sequence on A100.",
    )
    parser.add_argument("--hardware", default="a100")
    args, extra_args = parser.parse_known_args()

    os.chdir(REPO_ROOT)
    _check_inputs()

    print(f"Starting baseline sequence on hardware={args.hardware}")
    print(f"Extra run_training.py args: {extra_args}")

    for config in CONFIGS:
        _run_config(config, args.hardware, extra_args)

    print()
    print("All baseline runs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
