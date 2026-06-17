from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.datasets import build_dataloaders
from src.eval import evaluate
from src.models import build_model
from src.train import _load_config, _model_kwargs


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_eval_paths(
    run_dir: str | Path,
    config: str | Path,
    checkpoint: str | Path | None,
    output: str | Path | None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "run_dir": run_dir,
        "config": Path(config),
        "checkpoint": Path(checkpoint) if checkpoint else run_dir / "best.pt",
        "output": Path(output) if output else run_dir / "test_metrics.json",
    }


def evaluate_run(
    run_dir: str | Path,
    config: str | Path,
    checkpoint: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, float]:
    os.chdir(REPO_ROOT)
    paths = resolve_eval_paths(run_dir, config, checkpoint, output)
    if not paths["checkpoint"].exists():
        raise FileNotFoundError(f"Missing checkpoint: {paths['checkpoint']}")

    config_data = _load_config(paths["config"])
    device = torch.device(config_data["train"].get("device", "cpu"))
    _, test_loader, class_names = build_dataloaders(
        train_dir=config_data["data"]["train_dir"],
        val_dir=config_data["data"]["val_dir"],
        image_size=config_data["data"]["image_size"],
        batch_size=config_data["train"]["batch_size"],
        num_workers=config_data["data"].get("num_workers", 4),
        augment=config_data["data"].get("augment", "basic"),
    )
    model = build_model(
        architecture=config_data["model"]["architecture"],
        num_classes=len(class_names),
        pretrained=False,
        use_coord_attention=config_data["model"].get("use_coord_attention", False),
        **_model_kwargs(config_data["model"]),
    )
    checkpoint_data = torch.load(paths["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint_data["model_state"])
    model.to(device)

    metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes=len(class_names),
        tta_horizontal_flip=config_data.get("eval", {}).get("tta_horizontal_flip", False),
    )
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run directory.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_run(
        run_dir=args.run_dir,
        config=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
