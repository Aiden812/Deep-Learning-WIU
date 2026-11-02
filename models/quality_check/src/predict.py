"""Image quality gate — run before the classifier / detector.

Returns {"ok": bool, "reasons": [...], "scores": {...}}, e.g. reasons = ["blurry", "too_dark", "not_intraoral"].
"""

from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "best.pt"


def load_model(weights: Path = DEFAULT_WEIGHTS):
    raise NotImplementedError


def predict(image_path: Path, model=None) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    print(predict(Path(sys.argv[1])))
