"""Verify that every raw dataset has been downloaded into data/raw/.

Usage:
    python models/yolo_caries/scripts/check_downloads.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"

EXPECTED = [
    "kaggle_oral",
    "zenodo_benchmark",
    "zenodo_full",
    "dentalai",
    "mendeley_noncaries",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    missing = []
    for name in EXPECTED:
        folder = RAW_DIR / name
        n = count_images(folder) if folder.exists() else 0
        status = "OK" if n else "MISSING"
        print(f"{status:8} {name:20} {n:>7} images")
        if not n:
            missing.append(name)
    if missing:
        print(f"\n{len(missing)} dataset(s) missing - see README.md for download links.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
