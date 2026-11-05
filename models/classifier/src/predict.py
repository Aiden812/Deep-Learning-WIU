"""Condition classifier inference.

Returns {"class_id", "class_name", "confidence", "probs"} using IDs from shared/conditions.yaml.

The model is a Keras EfficientNetB0 whose first layers rescale pixels itself, so it takes
raw 0-255 RGB. Do not divide by 255 here: that double-scales the input and wrecks predictions.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "oral_classifier.keras"
DEFAULT_CLASS_NAMES = WEIGHTS_DIR / "class_names.json"
CONDITIONS_FILE = ROOT / "shared" / "conditions.yaml"
IMG_SIZE = 224


def load_conditions(path: Path = CONDITIONS_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["conditions"]


class Classifier:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, class_names: Path = DEFAULT_CLASS_NAMES):
        import keras

        self.model = keras.models.load_model(weights, compile=False)
        labels = json.loads(class_names.read_text(encoding="utf-8"))
        # Keras orders classes by folder name (capitals first), which differs from the
        # shared IDs, so translate each model output index via source_label.
        by_label = {c["source_label"]: c for c in load_conditions()}
        missing = [l for l in labels if l not in by_label]
        if missing:
            raise ValueError(f"class_names.json labels not in conditions.yaml: {missing}")
        self.conditions = [by_label[l] for l in labels]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        import tensorflow as tf

        image = ImageOps.exif_transpose(image).convert("RGB")
        arr = np.asarray(image, dtype=np.float32)  # 0-255, as in training
        # Same resize as keras.utils.image_dataset_from_directory (bilinear, no crop).
        return tf.image.resize(arr, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy()

    def predict_images(self, images: list[Image.Image]) -> dict:
        """Average the class probabilities over one or more photos of the same mouth."""
        batch = np.stack([self.preprocess(im) for im in images])
        probs = self.model.predict(batch, verbose=0).mean(axis=0)
        top = int(probs.argmax())
        best = self.conditions[top]
        return {
            "class_id": best["id"],
            "class_name": best["name"],
            "confidence": float(probs[top]),
            "probs": {c["name"]: float(p) for c, p in zip(self.conditions, probs)},
        }


_default: Classifier | None = None


def load_model(weights: Path = DEFAULT_WEIGHTS) -> Classifier:
    return Classifier(weights)


def predict(image_path: Path, model: Classifier | None = None) -> dict:
    global _default
    if model is None:
        _default = _default or load_model()
        model = _default
    with Image.open(image_path) as im:
        return model.predict_images([im])


if __name__ == "__main__":
    import sys

    print(predict(Path(sys.argv[1])))
