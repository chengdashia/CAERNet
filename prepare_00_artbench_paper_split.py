from __future__ import annotations

import os
import sys
from pathlib import Path


CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]
if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from src.prepare_artbench_paper_split import prepare_split


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    prepare_split(
        source_test_dir=Path("classify/data/artbench10/test"),
        output_dir=Path("classify/data/artbench10_paper"),
        val_per_class=500,
        seed=42,
        overwrite=True,
    )
    print("Prepared randomized ArtBench-10 paper split at classify/data/artbench10_paper")
