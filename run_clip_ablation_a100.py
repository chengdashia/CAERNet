from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CAERNET_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_DIR.parents[1]
WEIGHT_PATH = Path("/home/kmyh/classify/models/open_clip/vit_b16_openai.bin")

CONFIGS = [
    CAERNET_DIR / "configs/experiments/clip_zeroshot_vit_b16.yaml",
    CAERNET_DIR / "configs/experiments/clip_adapter_vit_b16_no_energy.yaml",
    CAERNET_DIR / "configs/experiments/clip_adapter_vit_b16_no_contrastive.yaml",
    CAERNET_DIR / "configs/experiments/clip_adapter_vit_b16.yaml",
]

DATA_DIRS = [
    Path("classify/data/artbench10/train"),
    Path("classify/data/artbench10_paper/val"),
    Path("classify/data/artbench10/test"),
]


def _check_inputs():
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            f"CLIP weight file not found: {WEIGHT_PATH}\n"
            "Put vit_b16_openai.bin at this path before running."
        )
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
        description="Run the original CLIP ablation sequence on A100.",
    )
    parser.add_argument("--hardware", default="a100")
    args, extra_args = parser.parse_known_args()

    os.chdir(REPO_ROOT)
    _check_inputs()

    print(f"Starting CLIP ablation sequence on hardware={args.hardware}")
    print(f"Extra run_training.py args: {extra_args}")

    for config in CONFIGS:
        _run_config(config, args.hardware, extra_args)

    print()
    print("All CLIP ablation runs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
