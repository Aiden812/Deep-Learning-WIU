"""Caries detection inference.

Returns a list of detections: [{"class_id", "class_name", "confidence", "bbox_xyxy"}].
"""

from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "best.pt"


def load_model(weights: Path = DEFAULT_WEIGHTS):
    raise NotImplementedError


def predict(image_path: Path, model=None, conf: float = 0.25) -> list[dict]:
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    print(predict(Path(sys.argv[1])))
