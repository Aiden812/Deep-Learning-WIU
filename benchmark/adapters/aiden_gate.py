"""Aiden's gate (model.pt from scripts/Aiden/train.py): uses his own loader in models/classifier/src/multilabel.py
(Resize to img_size, ImageNet normalise, mean of photo + mirror logits, Platt calibration, saved threshold).
DINOv2 needs torch.hub to fetch the facebookresearch/dinov2 code the first time."""
import sys

import numpy as np
import torch

from common import REPO


class AidenGate:
    batchable = True

    def __init__(self, weights, **_):
        sys.path.insert(0, str(REPO / "models" / "classifier" / "src"))
        import multilabel
        self.ml = multilabel
        self.ck, self.nets, self.dev = multilabel.load_model(weights)
        self.tf = multilabel.eval_tf(self.ck["img_size"])
        self.a, self.b = np.array(self.ck["calib_a"]), np.array(self.ck["calib_b"])
        self.thr = float(self.ck["thresholds"][0])

    def predict_batch(self, ims):
        x = torch.stack([self.tf(im) for im in ims]).to(self.dev)
        z = self.ml.ensemble_logits(self.nets, x)[:, 0].cpu().numpy()
        p = 1 / (1 + np.exp(-(self.a[0] * z + self.b[0])))
        return [dict(p_problem=float(v), flag_problem=bool(v >= self.thr), p={}, flag={}, boxes=None) for v in p]
