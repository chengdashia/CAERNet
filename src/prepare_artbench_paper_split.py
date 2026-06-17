from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def _link_or_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_split(
    source_test_dir: Path,
    output_dir: Path,
    val_per_class: int,
    seed: int = 42,
    overwrite: bool = False,
):
    if not source_test_dir.exists():
        raise FileNotFoundError(f"Missing source test directory: {source_test_dir}")

    if overwrite:
        for split_name in ("val", "test"):
            split_dir = output_dir / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)

    manifest = {
        "source_test_dir": str(source_test_dir),
        "output_dir": str(output_dir),
        "val_per_class": val_per_class,
        "seed": seed,
        "split_policy": "deterministic random shuffle per class",
        "classes": {},
    }

    for class_dir in sorted(path for path in source_test_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if len(images) <= val_per_class:
            raise ValueError(
                f"Class {class_dir.name} has {len(images)} images, "
                f"which is not enough for val_per_class={val_per_class}."
            )

        class_rng = random.Random(f"{seed}:{class_dir.name}")
        class_rng.shuffle(images)
        val_images = images[:val_per_class]
        test_images = images[val_per_class:]

        for image in val_images:
            _link_or_copy(image, output_dir / "val" / class_dir.name / image.name)
        for image in test_images:
            _link_or_copy(image, output_dir / "test" / class_dir.name / image.name)

        manifest["classes"][class_dir.name] = {
            "source_count": len(images),
            "val_count": len(val_images),
            "test_count": len(test_images),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a paper-style ArtBench-10 val/test split."
    )
    parser.add_argument(
        "--source-test-dir",
        type=Path,
        default=Path("classify/data/artbench10/test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("classify/data/artbench10_paper"),
    )
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_split(
        source_test_dir=args.source_test_dir,
        output_dir=args.output_dir,
        val_per_class=args.val_per_class,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
