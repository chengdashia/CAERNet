from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]
if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from src.datasets import build_dataloaders
from src.eval import evaluate
from src.models import build_model
from src.train import _load_config, _model_kwargs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a config split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config = _load_config(args.config)
    device = torch.device(config["train"].get("device", "cpu"))
    _, test_loader, class_names = build_dataloaders(
        train_dir=config["data"]["train_dir"],
        val_dir=config["data"]["val_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        augment=config["data"].get("augment", "basic"),
    )
    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=len(class_names),
        pretrained=False,
        use_coord_attention=config["model"].get("use_coord_attention", False),
        **_model_kwargs(config["model"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    metrics = evaluate(model, test_loader, device, num_classes=len(class_names))
    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
