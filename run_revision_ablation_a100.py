from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CAERNET_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_DIR.parents[1]
WEIGHT_PATH = Path("/home/kmyh/classify/models/open_clip/vit_b16_openai.bin")
REVISION_CONFIG_DIR = CAERNET_DIR / "configs" / "revision"

CONFIG_GROUPS = {
    "adaptation": [
        REVISION_CONFIG_DIR / "adaptation_linear_probe.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_only.yaml",
        REVISION_CONFIG_DIR / "adaptation_partial_only_b2.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_partial_b2.yaml",
        REVISION_CONFIG_DIR / "adaptation_full_visual.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_full_visual.yaml",
    ],
    "depth": [
        REVISION_CONFIG_DIR / "adaptation_adapter_only.yaml",
        REVISION_CONFIG_DIR / "depth_b1.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_partial_b2.yaml",
        REVISION_CONFIG_DIR / "depth_b4.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_full_visual.yaml",
    ],
    "regularizers": [
        REVISION_CONFIG_DIR / "regularizer_plain_ce.yaml",
        REVISION_CONFIG_DIR / "regularizer_label_smoothing.yaml",
        REVISION_CONFIG_DIR / "regularizer_mixup.yaml",
        REVISION_CONFIG_DIR / "regularizer_supcon.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_partial_b2.yaml",
    ],
    "clip_adapter": [
        REVISION_CONFIG_DIR / "clip_adapter_baseline.yaml",
    ],
    "low_data": [
        REVISION_CONFIG_DIR / "low_data_01.yaml",
        REVISION_CONFIG_DIR / "low_data_05.yaml",
        REVISION_CONFIG_DIR / "low_data_10.yaml",
        REVISION_CONFIG_DIR / "low_data_25.yaml",
    ],
}

DATA_DIRS = [
    Path("classify/data/artbench10/train"),
    Path("classify/data/artbench10_paper/val"),
    Path("classify/data/artbench10/test"),
]


def _check_inputs(configs: list[Path]) -> None:
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
    for config in configs:
        if not config.is_file():
            raise FileNotFoundError(f"Experiment config not found: {config}")


def _selected_configs(groups: list[str]) -> list[Path]:
    configs: list[Path] = []
    for group in groups:
        for config in CONFIG_GROUPS[group]:
            if config not in configs:
                configs.append(config)
    return configs


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
    print(f"Group output suffix: seed{seed}")
    print("=" * 72)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reviewer-requested CLIP ablations on A100.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[*CONFIG_GROUPS, "all"],
        default=["all"],
        help="Experiment groups. Default: all.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[41, 42, 43],
        help="Random seeds. Default: 41 42 43.",
    )
    parser.add_argument("--hardware", default="a100")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()

    groups = list(CONFIG_GROUPS) if "all" in args.groups else args.groups
    configs = _selected_configs(groups)

    os.chdir(REPO_ROOT)
    _check_inputs(configs)
    print(f"Starting revision experiments on hardware={args.hardware}")
    print(f"Groups: {groups}")
    print(f"Seeds: {args.seeds}")
    print(f"Unique configs: {len(configs)}")

    for seed in args.seeds:
        for config in configs:
            _run_config(config, seed, args.hardware, args.dry_run, extra_args)

    print()
    print("All requested revision ablations finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
