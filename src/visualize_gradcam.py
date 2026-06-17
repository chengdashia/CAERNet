import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM placeholder for paper figures.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise NotImplementedError(
        "Grad-CAM visualization is planned after the training baseline is stable. "
        f"Received checkpoint={args.checkpoint}, image={args.image}, output={args.output}."
    )


if __name__ == "__main__":
    main()
