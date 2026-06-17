from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets import build_imagefolder_dataset
from src.metrics import confusion_matrix_array
from src.models import build_model
from src.train import _load_config, _model_kwargs


REPO_ROOT = Path(__file__).resolve().parents[3]


def _to_number(value: str) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_history(history_path: str | Path) -> list[dict[str, Any]]:
    """Read a training history CSV and convert numeric fields to floats."""
    with Path(history_path).open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: _to_number(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    numeric_rows = [row for row in rows if isinstance(row.get(metric), float)]
    if not numeric_rows:
        return None
    return max(numeric_rows, key=lambda row: row[metric])


def summarize_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize convergence, best epochs, and train/validation gap."""
    if not rows:
        return {"epochs": 0}

    last = rows[-1]
    best_val = _best_row(rows, "val_accuracy")
    best_test = _best_row(rows, "test_tta_accuracy") or _best_row(rows, "test_accuracy")
    min_val_loss = min(
        (row for row in rows if isinstance(row.get("val_loss"), float)),
        key=lambda row: row["val_loss"],
        default=None,
    )

    summary: dict[str, Any] = {
        "epochs": int(last.get("epoch", len(rows))),
        "last_epoch": int(last.get("epoch", len(rows))),
        "last_train_accuracy": last.get("train_accuracy"),
        "last_val_accuracy": last.get("val_accuracy"),
        "last_val_loss": last.get("val_loss"),
    }

    if best_val is not None:
        summary.update(
            {
                "best_val_epoch": int(best_val["epoch"]),
                "best_val_accuracy": best_val["val_accuracy"],
                "train_accuracy_at_best_val": best_val.get("train_accuracy"),
                "generalization_gap_at_best_val": (
                    best_val.get("train_accuracy") - best_val["val_accuracy"]
                    if isinstance(best_val.get("train_accuracy"), float)
                    else None
                ),
            }
        )

    if best_test is not None:
        metric_name = "test_tta_accuracy" if "test_tta_accuracy" in best_test else "test_accuracy"
        summary.update(
            {
                "best_test_metric": metric_name,
                "best_test_epoch": int(best_test["epoch"]),
                "best_test_accuracy": best_test[metric_name],
            }
        )

    if min_val_loss is not None:
        summary.update(
            {
                "min_val_loss_epoch": int(min_val_loss["epoch"]),
                "min_val_loss": min_val_loss["val_loss"],
            }
        )

    return summary


def class_counts(data_dir: str | Path) -> dict[str, int]:
    """Count images per ImageFolder class without loading image tensors."""
    dataset = datasets.ImageFolder(root=str(Path(data_dir)))
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, class_index in dataset.samples:
        counts[dataset.classes[class_index]] += 1
    return counts


@torch.no_grad()
def collect_predictions(
    model,
    dataloader: DataLoader,
    device: torch.device,
    tta_horizontal_flip: bool = False,
) -> tuple[list[int], list[int]]:
    """Return true and predicted labels for confusion-matrix diagnostics."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets in dataloader:
        images = images.to(device)
        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        if tta_horizontal_flip:
            flipped_output = model(torch.flip(images, dims=(-1,)))
            flipped_logits = flipped_output[0] if isinstance(flipped_output, tuple) else flipped_output
            logits = (logits + flipped_logits) * 0.5

        y_true.extend(targets.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return y_true, y_pred


def per_class_metrics(matrix, class_names: list[str]) -> list[dict[str, Any]]:
    """Compute one-vs-rest precision, recall, and F1 from a confusion matrix."""
    rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "class": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": true_positive,
                "top_confusion": _top_confusion(matrix, class_names, index),
            }
        )
    return rows


def _top_confusion(matrix, class_names: list[str], class_index: int) -> str:
    row = matrix[class_index].copy()
    row[class_index] = 0
    predicted_index = int(row.argmax())
    count = int(row[predicted_index])
    if count == 0:
        return ""
    return f"{class_names[predicted_index]}:{count}"


def write_table(rows: list[dict[str, Any]], path: str | Path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(matrix, class_names: list[str], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def run_diagnostics(
    run_dir: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate history, dataset, and checkpoint diagnostics for one run."""
    os.chdir(REPO_ROOT)
    run_dir = Path(run_dir)
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else run_dir / "best.pt"
    output_dir = Path(output_dir) if output_dir else run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path)
    history_path = run_dir / "history.csv"
    history_summary = summarize_history(read_history(history_path)) if history_path.exists() else {}

    data_summary = {
        "train": class_counts(config["data"]["train_dir"]),
        "val": class_counts(config["data"]["val_dir"]),
    }
    eval_config = config.get("eval", {})
    if eval_config.get("test_dir"):
        data_summary["test"] = class_counts(eval_config["test_dir"])

    device = torch.device(config["train"].get("device", "cpu"))
    train_dataset = datasets.ImageFolder(root=str(Path(config["data"]["train_dir"])))
    class_names = train_dataset.classes

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=len(class_names),
        pretrained=False,
        use_coord_attention=config["model"].get("use_coord_attention", False),
        **_model_kwargs(config["model"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    split_summaries: dict[str, Any] = {}
    for split_name, split_dir in (
        ("val", config["data"]["val_dir"]),
        ("test", eval_config.get("test_dir")),
    ):
        if not split_dir:
            continue
        dataset = build_imagefolder_dataset(split_dir, config["data"]["image_size"], train=False)
        if dataset.classes != class_names:
            raise ValueError(
                "Class folders must match. "
                f"train={class_names}, {split_name}={dataset.classes}"
            )
        loader = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 4),
            pin_memory=True,
            prefetch_factor=2 if config["data"].get("num_workers", 4) > 0 else None,
            persistent_workers=config["data"].get("num_workers", 4) > 0,
        )
        y_true, y_pred = collect_predictions(
            model,
            loader,
            device=device,
            tta_horizontal_flip=eval_config.get("tta_horizontal_flip", False),
        )
        matrix = confusion_matrix_array(y_true, y_pred, num_classes=len(class_names))
        per_class = per_class_metrics(matrix, class_names)
        write_confusion_matrix(matrix, class_names, output_dir / f"{split_name}_confusion_matrix.csv")
        write_table(per_class, output_dir / f"{split_name}_per_class_metrics.csv")
        split_summaries[split_name] = {
            "accuracy": sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1),
            "worst_classes_by_f1": sorted(per_class, key=lambda row: row["f1"])[:3],
        }

    summary = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "history": history_summary,
        "class_counts": data_summary,
        "splits": split_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
