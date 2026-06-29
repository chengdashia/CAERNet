from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/kmyh/classify/third_party")
DATA_ROOT = Path("/home/kmyh/classify/data/artbench10")

BASELINE_REPOS = {
    "coop": REPO_ROOT / "CoOp",
    "cocoop": REPO_ROOT / "CoOp",
    "maple": REPO_ROOT / "multimodal-prompt-learning",
    "promptsrc": REPO_ROOT / "PromptSRC",
    "tip_adapter": REPO_ROOT / "Tip-Adapter",
}

OFFICIAL_URLS = {
    "coop": "https://github.com/KaiyangZhou/CoOp",
    "cocoop": "https://github.com/KaiyangZhou/CoOp",
    "maple": "https://github.com/muzairkhattak/multimodal-prompt-learning",
    "promptsrc": "https://github.com/muzairkhattak/PromptSRC",
    "tip_adapter": "https://github.com/gaopengcuhk/Tip-Adapter",
}


def _command(method: str, seed: int, shots: int) -> list[str]:
    repo = BASELINE_REPOS[method]
    python = os.environ.get("PYTHON_BIN", sys.executable)
    if method in {"coop", "cocoop"}:
        trainer = "CoOp" if method == "coop" else "CoCoOp"
        return [
            python,
            str(repo / "train.py"),
            "--root",
            str(DATA_ROOT),
            "--seed",
            str(seed),
            "--trainer",
            trainer,
            "--dataset-config-file",
            str(repo / "configs/datasets/artbench10.yaml"),
            "--config-file",
            str(repo / f"configs/trainers/{trainer}/vit_b16.yaml"),
            "DATASET.NUM_SHOTS",
            str(shots),
        ]
    if method in {"maple", "promptsrc"}:
        trainer = "MaPLe" if method == "maple" else "PromptSRC"
        return [
            python,
            str(repo / "train.py"),
            "--root",
            str(DATA_ROOT),
            "--seed",
            str(seed),
            "--trainer",
            trainer,
            "--dataset-config-file",
            str(repo / "configs/datasets/artbench10.yaml"),
            "--config-file",
            str(repo / f"configs/trainers/{trainer}/vit_b16.yaml"),
            "DATASET.NUM_SHOTS",
            str(shots),
        ]
    return [
        python,
        str(repo / "main.py"),
        "--config",
        str(repo / "configs/artbench10.yaml"),
        "--seed",
        str(seed),
        "--shots",
        str(shots),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute official CoOp/CoCoOp/MaPLe/PromptSRC/"
            "Tip-Adapter commands on A100."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(BASELINE_REPOS),
        default=list(BASELINE_REPOS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands. Without this flag, only print the plan.",
    )
    args = parser.parse_args()

    for method in args.methods:
        repo = BASELINE_REPOS[method]
        if not repo.is_dir():
            print(f"{method}: missing {repo}")
            print(f"  Clone official repository: {OFFICIAL_URLS[method]}")
            if args.execute:
                raise FileNotFoundError(repo)
            continue
        for seed in args.seeds:
            for shots in args.shots:
                command = _command(method, seed, shots)
                print(" ".join(command))
                if args.execute:
                    subprocess.run(command, cwd=repo, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
