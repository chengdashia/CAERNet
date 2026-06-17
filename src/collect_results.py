from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "name",
    "run_dir",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "best_val_precision",
    "best_val_recall",
    "best_val_ece",
    "test_accuracy",
    "test_macro_f1",
    "test_precision",
    "test_recall",
    "test_ece",
]


def _to_number(value: str | None) -> float | int | None:
    if value is None or value == "":
        return None
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _read_history_best(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        return {}

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    best = max(rows, key=lambda row: float(row.get("val_accuracy") or 0.0))
    return {
        "best_epoch": _to_number(best.get("epoch")),
        "best_val_accuracy": _to_number(best.get("val_accuracy")),
        "best_val_macro_f1": _to_number(best.get("val_macro_f1")),
        "best_val_precision": _to_number(best.get("val_precision")),
        "best_val_recall": _to_number(best.get("val_recall")),
        "best_val_ece": _to_number(best.get("val_ece")),
    }


def _read_test_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_precision": metrics.get("precision"),
        "test_recall": metrics.get("recall"),
        "test_ece": metrics.get("ece"),
    }


def collect_run_result(name: str, run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    row: dict[str, Any] = {
        "name": name,
        "run_dir": str(run_dir),
    }
    row.update(_read_history_best(run_dir / "history.csv"))
    row.update(_read_test_metrics(run_dir / "test_metrics.json"))
    return {column: row.get(column, "") for column in OUTPUT_COLUMNS}


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, Path(path)
    path = Path(spec)
    return path.name, path


def _write_csv(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in OUTPUT_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in OUTPUT_COLUMNS) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect CAERNet run metrics into one table.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as name=path or just path. Can be repeated.",
    )
    parser.add_argument("--csv", type=Path, default=Path("classify/outputs/tables/results.csv"))
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    rows = [collect_run_result(*_parse_run_spec(spec)) for spec in args.run]
    _write_csv(rows, args.csv)
    print(f"Wrote CSV: {args.csv}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown_table(rows) + "\n", encoding="utf-8")
        print(f"Wrote Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
