from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CAERNET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAERNET_ROOT.parents[1]
if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from src.diagnostics import run_diagnostics


def _resolve_cli_path(path: Path | None) -> Path | None:
    """Resolve paths whether the script is run from repo root or CAERNet-A100."""
    if path is None or path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, CAERNET_ROOT / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return path


def main():
    parser = argparse.ArgumentParser(description="Diagnose a trained CAERNet run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_diagnostics(
        run_dir=_resolve_cli_path(args.run_dir),
        config_path=_resolve_cli_path(args.config),
        checkpoint_path=_resolve_cli_path(args.checkpoint),
        output_dir=_resolve_cli_path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
