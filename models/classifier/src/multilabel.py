"""Aiden's classifier (stage-1 gate since 2026-11-12; earlier multi-label runs): model definition and inference.

predict(image) -> {"conditions": [{"name", "probability", "threshold", "flagged"}, ...]} for the checkpoint written by
scripts/Aiden/train.py (one or more models averaged, + per-condition calibration and thresholds).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "results" / "Aiden" / "dinov2_vitb14" / "model.pt"
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def eval_tf(img):
    return transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(), NORM])


class Net(nn.Module):
    def __init__(self, arch, n_classes, pretrained=True):
        super().__init__()
        self.arch = arch
        if arch == "densenet121":
            self.body = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
            dim = self.body.classifier.in_features
            self.body.classifier = nn.Identity()
        else:   # dinov2_vits14 / dinov2_vitb14
            self.body = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)
            dim = self.body.embed_dim * 2
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        if self.arch == "densenet121":
            return self.head(self.body(x))
        t = self.body.forward_features(x)
        return self.head(torch.cat([t["x_norm_clstoken"], t["x_norm_patchtokens"].mean(1)], 1))


@torch.no_grad()
def ensemble_logits(nets, x):
    """Mean logit over the fold models and a horizontal flip."""
    with torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
        return torch.stack([m(v) for m in nets for v in (x, x.flip(3))]).float().mean(0)


def load_model(weights=DEFAULT_WEIGHTS, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    nets = []
    for sd in ck.pop("state_dicts"):
        m = Net(ck["arch"], len(ck["classes"]), pretrained=False)
        m.load_state_dict(sd)
        nets.append(m.float().eval().to(device))
    return ck, nets, device


def predict(image, model=None):
    """image: path or PIL image. Probabilities are calibrated on out-of-fold predictions."""
    ck, nets, device = model or load_model()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    z = ensemble_logits(nets, eval_tf(ck["img_size"])(img.convert("RGB")).unsqueeze(0).to(device))[0].cpu().numpy()
    p = 1 / (1 + np.exp(-(np.array(ck["calib_a"]) * z + np.array(ck["calib_b"]))))
    return {"conditions": [dict(name=c, probability=float(pi), threshold=float(t), flagged=bool(pi >= t))
                           for c, pi, t in zip(ck["classes"], p, ck["thresholds"])]}


if __name__ == "__main__":
    import sys

    print(predict(Path(sys.argv[1])))
