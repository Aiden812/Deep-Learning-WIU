"""Deduplicate images across sources (perceptual hash) and write train/val/test splits.

Outputs:
    data/processed/yolo/{images,labels}/{train,val,test}/
    manifests/yolo_split.csv   (path, source, sha256, phash, split)

Usage:
    python models/yolo_caries/scripts/dedup_and_split.py --seed 42
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, nargs="+", default=[Path("data/interim")])
    parser.add_argument("--dst", type=Path, default=Path("data/processed/yolo"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/yolo_split.csv"))
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
