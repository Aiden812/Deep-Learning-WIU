"""The team's Keras EfficientNetB0 6-class baseline (models/classifier/src/predict.py): raw 0-255 RGB,
bilinear squash to 224, softmax over 6 classes incl. Normal. p_problem = 1 - p(Normal); verdict = argmax."""
import json

import numpy as np

from . import to_ours


class Keras6:
    batchable = True

    def __init__(self, weights, **_):
        import keras
        import tensorflow as tf
        self.tf = tf
        self.model = keras.models.load_model(weights, compile=False)
        self.labels = json.loads((weights.parent / "class_names.json").read_text(encoding="utf-8"))
        self.normal = self.labels.index("Normal")

    def predict_batch(self, ims):
        x = np.stack([self.tf.image.resize(np.asarray(im, dtype=np.float32), (224, 224), method="bilinear").numpy()
                      for im in ims])
        out = []
        for pr in self.model.predict(x, verbose=0):
            k = int(pr.argmax())
            out.append(dict(p_problem=float(1 - pr[self.normal]), flag_problem=k != self.normal,
                            p={to_ours(c): float(pr[i]) for i, c in enumerate(self.labels) if to_ours(c)},
                            flag={to_ours(c): k == i for i, c in enumerate(self.labels) if to_ours(c)}, boxes=None))
        return out
