from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.datasets import build_dataloaders
from src.metrics import classification_metrics, expected_calibration_error
from src.models import build_model


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    num_classes: int,
    tta_horizontal_flip: bool = False,
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    all_probabilities = []
    all_targets = []
    total_loss = 0.0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        if tta_horizontal_flip:
            flipped_output = model(torch.flip(images, dims=(-1,)))
            flipped_logits = flipped_output[0] if isinstance(flipped_output, tuple) else flipped_output
            logits = (logits + flipped_logits) * 0.5
        loss = torch.nn.functional.cross_entropy(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += float(loss.cpu()) * batch_size
        total_examples += batch_size
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        all_probabilities.append(probabilities.cpu())
        all_targets.append(targets.cpu())

    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_examples, 1)
    if all_probabilities:
        metrics["ece"] = expected_calibration_error(
            torch.cat(all_probabilities),
            torch.cat(all_targets),
        )
    return metrics


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _model_kwargs(model_config: dict) -> dict:
    ignored = {"architecture", "pretrained", "use_coord_attention"}
    return {key: value for key, value in model_config.items() if key not in ignored}


def main():
    parser = argparse.ArgumentParser(description="Evaluate an art classification checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    device = torch.device(config["train"].get("device", "cpu"))
    _, val_loader, class_names = build_dataloaders(
        train_dir=config["data"]["train_dir"],
        val_dir=config["data"]["val_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        normalize=config["data"].get("normalize", "imagenet"),
    )
    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=len(class_names),
        class_names=class_names,
        pretrained=False,
        use_coord_attention=config["model"].get("use_coord_attention", False),
        **_model_kwargs(config["model"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    metrics = evaluate(
        model,
        val_loader,
        device,
        num_classes=len(class_names),
        tta_horizontal_flip=config.get("eval", {}).get("tta_horizontal_flip", False),
    )
    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
