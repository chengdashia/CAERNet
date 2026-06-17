from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.models import build_model


def summarize_model(model: torch.nn.Module) -> dict[str, int | float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": round(total_params / 1_000_000, 4),
        "trainable_params_m": round(trainable_params / 1_000_000, 4),
    }


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _model_kwargs(model_config: dict) -> dict:
    ignored = {"architecture", "pretrained", "use_coord_attention"}
    return {key: value for key, value in model_config.items() if key not in ignored}


def main():
    parser = argparse.ArgumentParser(description="Summarize a configured CAERNet model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=args.num_classes,
        pretrained=False,
        use_coord_attention=config["model"].get("use_coord_attention", False),
        **_model_kwargs(config["model"]),
    )
    summary = {
        "architecture": config["model"]["architecture"],
        **summarize_model(model),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
