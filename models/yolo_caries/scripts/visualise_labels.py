"""Draw YOLO labels on a random sample of images for visual QA.

Usage:
    python models/yolo_caries/scripts/visualise_labels.py --split train --n 16
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/yolo"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None, help="save grid to file instead of showing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
