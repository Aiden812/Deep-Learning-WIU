"""Convert DentalAI semantic-segmentation masks to YOLO bounding-box labels.

DentalAI (Roboflow export) ships `<image>.jpg` + `<image>_mask.png`, where pixel
values are: 0 background, 1 Caries, 2 Cavity, 3 Crack, 4 Tooth.
Each connected component of a target class becomes one YOLO box.

Usage:
    python models/yolo_caries/scripts/convert_dentalai_to_yolo.py \
        --src data/raw/dentalai --dst data/interim/dentalai_yolo
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/dentalai"))
    parser.add_argument("--dst", type=Path, default=Path("data/interim/dentalai_yolo"))
    parser.add_argument("--min-area", type=int, default=20, help="drop components smaller than this (px)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
