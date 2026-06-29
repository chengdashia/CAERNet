from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


CAERNET_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_DIR.parents[1]
WEIGHT_PATH = Path("/home/kmyh/classify/models/open_clip/vit_b16_openai.bin")
CONFIG_DIR = CAERNET_DIR / "configs" / "mlssa"

REVISION_CONFIG_DIR = CAERNET_DIR / "configs" / "revision"

CONFIG_GROUPS = {
    "comparison": [
        REVISION_CONFIG_DIR / "adaptation_linear_probe.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_only.yaml",
        REVISION_CONFIG_DIR / "adaptation_partial_only_b2.yaml",
        REVISION_CONFIG_DIR / "adaptation_adapter_partial_b2.yaml",
        REVISION_CONFIG_DIR / "adaptation_full_visual.yaml",
        REVISION_CONFIG_DIR / "clip_adapter_baseline.yaml",
        CONFIG_DIR / "clip_mlssa_frozen.yaml",
        CONFIG_DIR / "clip_mlssa_full.yaml",
    ],
    "components": [
        REVISION_CONFIG_DIR / "adaptation_partial_only_b2.yaml",
        CONFIG_DIR / "clip_mlssa_single_final_mean_only.yaml",
        CONFIG_DIR / "clip_mlssa_single_final.yaml",
        CONFIG_DIR / "clip_mlssa_uniform_fusion.yaml",
        CONFIG_DIR / "clip_mlssa_no_gate.yaml",
        CONFIG_DIR / "clip_mlssa_no_orth.yaml",
        CONFIG_DIR / "clip_mlssa_full.yaml",
    ],
    "layers": [
        CONFIG_DIR / "clip_mlssa_single_final.yaml",
        CONFIG_DIR / "clip_mlssa_layers_8_12.yaml",
        CONFIG_DIR / "clip_mlssa_full.yaml",
        CONFIG_DIR / "clip_mlssa_layers_even.yaml",
    ],
}


def _selected_configs(groups: list[str]) -> list[Path]:
    configs: list[Path] = []
    for group in groups:
        for config in CONFIG_GROUPS[group]:
            if config not in configs:
                configs.append(config)
    return configs


def _check_inputs(configs: list[Path]) -> None:
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"CLIP weight file not found: {WEIGHT_PATH}")
    for path in (
        Path("classify/data/artbench10/train"),
        Path("classify/data/artbench10_paper/val"),
        Path("classify/data/artbench10/test"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {path}")
    for config in configs:
        if not config.is_file():
            raise FileNotFoundError(f"MLSSA config not found: {config}")


def _expected_run_dir(config: Path, seed: int, hardware: str) -> Path:
    hardware_path = Path(hardware)
    if not hardware_path.is_file():
        hardware_path = CAERNET_DIR / "configs" / "hardware" / f"{hardware}.yaml"
    hardware_config = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    run_root = Path(hardware_config["output"]["run_root"])
    run_name = f"{config_data['output']['run_name']}_seed{seed}"
    return REPO_ROOT / run_root / run_name


def _run_config(
    config: Path,
    seed: int,
    hardware: str,
    dry_run: bool,
    force: bool,
) -> None:
    run_dir = _expected_run_dir(config, seed, hardware)
    if not dry_run and run_dir.exists():
        if force:
            print(f"Removing existing run: {run_dir}")
            shutil.rmtree(run_dir)
        elif (run_dir / "test_metrics.json").is_file():
            print(f"Skipping completed run: {run_dir}")
            return
        else:
            raise RuntimeError(
                f"Incomplete run directory exists: {run_dir}. "
                "Resume manually or rerun with --force."
            )

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
        description="Run controlled MLSSA experiments sequentially on A100.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[*CONFIG_GROUPS, "all"],
        default=["all"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--hardware", default="a100")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    groups = list(CONFIG_GROUPS) if "all" in args.groups else args.groups
    configs = _selected_configs(groups)
    os.chdir(REPO_ROOT)
    _check_inputs(configs)

    print(f"MLSSA groups: {groups}")
    print(f"Seeds: {args.seeds}")
    print(f"Unique configs: {len(configs)}")
    for seed in args.seeds:
        for config in configs:
            _run_config(config, seed, args.hardware, args.dry_run, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
