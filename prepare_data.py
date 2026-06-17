"""Prepare ArtBench-10 dataset: extract tar and create train/val/test split.

Pipeline:
    1. Extract artbench-10-imagefolder-split.tar → artbench10/{train,test}
    2. Split val from TRAIN (not test!) → artbench10_paper/val
    3. Keep full original test set → artbench10/test (10K, for fair evaluation)

This ensures:
    - Train: 4500/class × 10 = 45,000
    - Val:   500/class × 10 = 5,000  (from train, for model selection)
    - Test:  1000/class × 10 = 10,000 (full original, for final evaluation)

Usage:
    cd /Users/dong/Documents/SCI
    python classify/CAERNet/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tarfile
from pathlib import Path

CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]

# Ensure we work from the repository root so relative paths resolve correctly.
os.chdir(REPO_ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _link_or_copy(source: Path, target: Path):
    """Hard-link if possible, otherwise copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _count_class_images(class_dir: Path) -> int:
    """Count image files directly inside a class directory (non-recursive)."""
    count = 0
    with os.scandir(class_dir) as entries:
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES:
                count += 1
    return count


def extract_tar(tar_path: Path, output_dir: Path):
    """Extract the ArtBench-10 tar archive."""
    print(f"Extracting {tar_path} ...")
    print("  (This may take several minutes for a large archive)")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r:*") as tar:
        tar.extractall(path=str(output_dir))
    print(f"  Extracted to: {output_dir}")


def find_train_test(root: Path) -> tuple[Path | None, Path | None]:
    """Locate train/ and test/ directories, checking one level of nesting."""
    train = root / "train"
    test = root / "test"
    if train.is_dir() and test.is_dir():
        return train, test
    # Check one level deeper (tar may wrap in a parent directory)
    for child in sorted(root.iterdir()):
        if child.is_dir():
            t = child / "train"
            e = child / "test"
            if t.is_dir() and e.is_dir():
                return t, e
    return None, None


def _move_to_root(root: Path):
    """Move train/test from nested directory to root level."""
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in {"train", "test"}:
            continue
        for name in ("train", "test"):
            src = child / name
            dst = root / name
            if src.is_dir() and not dst.exists():
                shutil.move(str(src), str(dst))
        try:
            child.rmdir()
        except OSError:
            pass


def split_val_from_train(
    train_dir: Path,
    val_dir: Path,
    val_per_class: int,
    seed: int = 42,
):
    """Create a validation split from the training set.

    For each class:
        - Deterministic shuffle with per-class seed
        - **Move** first ``val_per_class`` images out of train into val_dir
        - Remaining images stay in train_dir
    """
    val_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "source": str(train_dir),
        "val_dir": str(val_dir),
        "val_per_class": val_per_class,
        "seed": seed,
        "classes": {},
    }

    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(images) <= val_per_class:
            raise ValueError(
                f"Class '{class_dir.name}': {len(images)} images is not enough "
                f"for val_per_class={val_per_class} (need strictly more)."
            )

        rng = random.Random(f"{seed}:{class_dir.name}")
        rng.shuffle(images)

        val_images = images[:val_per_class]
        remaining = len(images) - val_per_class

        for img in val_images:
            dst = val_dir / class_dir.name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), str(dst))

        manifest["classes"][class_dir.name] = {
            "original_count": len(images),
            "val_count": len(val_images),
            "remaining_train": remaining,
        }

    (val_dir.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract ArtBench-10 tar and prepare train/val/test split.",
    )
    parser.add_argument(
        "--tar",
        type=Path,
        default=Path("classify/data/artbench10_raw/artbench-10-imagefolder-split.tar"),
        help="Path to the ArtBench-10 tar archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("classify/data/artbench10"),
        help="Directory to extract train/ and test/ into.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("classify/data/artbench10_paper/val"),
        help="Output path for the validation split.",
    )
    parser.add_argument(
        "--val-per-class", type=int, default=500,
        help="Number of validation images per class (taken from train).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remove existing directories before creating.",
    )
    args = parser.parse_args()

    # --- Step 1: Extract tar ---
    train_dir = args.output / "train"
    test_dir = args.output / "test"

    paper_root = args.val_dir.parent
    if args.overwrite:
        for path in (args.output, paper_root):
            if path.exists():
                print(f"Removing existing directory: {path}")
                shutil.rmtree(path)

    if not train_dir.is_dir():
        if not args.tar.exists():
            raise FileNotFoundError(
                f"Tar file not found: {args.tar}\n"
                f"Download ArtBench-10 and place the tar at that path."
            )
        extract_tar(args.tar, args.output)
        found_train, found_test = find_train_test(args.output)
        if found_train is None:
            raise RuntimeError(
                f"Cannot find train/ and test/ directories after extraction. "
                f"Check the contents of {args.output}"
            )
        if found_train != train_dir:
            _move_to_root(args.output)
    else:
        print(f"  {train_dir} already exists, skipping extraction.")

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Check the tar archive structure."
        )
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            f"Check the tar archive structure."
        )

    # --- Step 2: Create val split from train ---
    if args.val_dir.is_dir():
        print(f"  Val already exists: {args.val_dir}, skipping split.")
    else:
        print(f"\nSplitting validation set from training data ...")
        split_val_from_train(
            train_dir=train_dir,
            val_dir=args.val_dir,
            val_per_class=args.val_per_class,
            seed=args.seed,
        )
        print(f"  Val split created: {args.val_dir}")

    # --- Summary ---
    print("\n" + "=" * 62)
    print("Dataset preparation complete!")
    print("=" * 62)
    print()
    print("Final directory layout:")
    print()

    for name, path in [
        ("Train", train_dir),
        ("Val  ", args.val_dir),
        ("Test ", test_dir),
    ]:
        if path.is_dir():
            classes = sorted(p.name for p in path.iterdir() if p.is_dir())
            counts = [_count_class_images(path / c) for c in classes]
            total = sum(counts)
            avg = total // max(len(counts), 1)
            print(f"  {name}: {path}")
            print(f"         {len(classes)} classes × ~{avg} images = {total} total")
            print()

    print("Recommended config paths:")
    print(f"  data.train_dir: {train_dir}")
    print(f"  data.val_dir:   {args.val_dir}")
    print(f"  eval.test_dir:  {test_dir}")


if __name__ == "__main__":
    main()
