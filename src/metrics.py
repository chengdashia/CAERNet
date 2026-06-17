from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred, num_classes: int) -> dict[str, float]:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(targets)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probabilities.device)
    ece = torch.zeros((), device=probabilities.device)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        proportion = in_bin.float().mean()
        if proportion.item() > 0:
            accuracy = accuracies[in_bin].float().mean()
            confidence = confidences[in_bin].mean()
            ece += torch.abs(confidence - accuracy) * proportion

    return float(ece.cpu())


def confusion_matrix_array(y_true, y_pred, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix
