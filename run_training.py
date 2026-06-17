from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import yaml


CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]

if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from src.train import train_from_config


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "hardware_tag":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_effective_config(config_path: Path, hardware_path: Path | None = None) -> dict:
    config = _load_yaml(config_path)
    if hardware_path is not None:
        hardware = _load_yaml(hardware_path)
        config = _deep_merge(config, hardware)
        run_name = config.get("output", {}).get("run_name")
        run_root = hardware.get("output", {}).get("run_root")
        if run_name and run_root:
            config.setdefault("output", {})["run_dir"] = str(Path(run_root) / run_name)
    elif config.get("output", {}).get("run_name") and "run_dir" not in config.get("output", {}):
        config.setdefault("output", {})["run_dir"] = str(
            Path("classify/outputs/runs") / config["output"]["run_name"]
        )
    return config


def _ensure_dataset_ready(config: dict):
    required_paths = [
        Path(config["data"]["train_dir"]),
        Path(config["data"]["val_dir"]),
    ]
    test_dir = config.get("eval", {}).get("test_dir")
    if test_dir:
        required_paths.append(Path(test_dir))

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Dataset is not ready. Missing paths:\n"
            f"{missing_text}\n\n"
            "Run this once from /home/kmyh/classify/CAERNet:\n"
            "  python prepare_data.py\n"
        )


def run(config_relative_path: str):
    config_path = CAERNET_ROOT / config_relative_path
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    os.chdir(REPO_ROOT)
    config = build_effective_config(config_path)
    _ensure_dataset_ready(config)
    train_from_config(config)


def _resolve_config_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    candidate = CAERNET_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing config file: {path}")


def _resolve_hardware_path(hardware: str | None) -> Path | None:
    if not hardware:
        return None
    hardware_path = Path(hardware)
    if hardware_path.exists():
        return hardware_path
    candidate = CAERNET_ROOT / "configs" / "hardware" / f"{hardware}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing hardware profile: {hardware}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ArtBench classification experiments.")
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    parser.add_argument("--hardware", default=None, help="Hardware profile name or YAML path.")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and run one forward pass.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config_path = _resolve_config_path(args.config)
    hardware_path = _resolve_hardware_path(args.hardware)
    config = build_effective_config(config_path, hardware_path)
    _ensure_dataset_ready(config)
    train_from_config(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
