"""Aiden's classifier (stage-1 gate since 2026-11-12; earlier multi-label runs): model definition and inference.

Used by train.py (Net, eval_tf, NORM) and by an app (load_model, predict).
predict(image) -> {"conditions": [{"name": "has_issue", "probability", "threshold", "flagged"}]} for the model.pt written by
scripts/Aiden/train.py (the trained weights + the calibration and threshold chosen on the val sets).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "results" / "Aiden" / "dinov2_vitb14" / "model.pt"
# Pretrained models expect pixels normalised with the ImageNet mean / standard deviation per colour channel
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# Preprocessing at prediction time: resize to img x img (224), to a tensor, normalise. No random augmentation here.
def eval_tf(img):
    return transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(), NORM])


# The model = pretrained backbone ("body", turns a photo into a feature vector) + one linear layer ("head", turns the
# features into one score per output). Only the head is new; the body starts from pretrained weights.
class Net(nn.Module):
    def __init__(self, arch, n_classes, pretrained=True):
        super().__init__()
        self.arch = arch
        if arch == "densenet121":
            self.body = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
            dim = self.body.classifier.in_features
            self.body.classifier = nn.Identity()   # drop ImageNet's 1000-class layer, keep the 1024 features
        else:   # dinov2_vits14 / dinov2_vitb14
            self.body = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)
            dim = self.body.embed_dim * 2   # 768 (CLS token) + 768 (average patch token) for ViT-B
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        if self.arch == "densenet121":
            return self.head(self.body(x))
        # ViT: the image is cut into 14x14-pixel patches. CLS token = summary of the whole image; the patch average
        # adds the local detail (small lesions). Both are joined and fed to the head.
        t = self.body.forward_features(x)
        return self.head(torch.cat([t["x_norm_clstoken"], t["x_norm_patchtokens"].mean(1)], 1))


@torch.no_grad()
def ensemble_logits(nets, x):
    """Mean raw score over the saved models (one for the gate) and the image + its mirror."""
    with torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
        return torch.stack([m(v) for m in nets for v in (x, x.flip(3))]).float().mean(0)


# Load model.pt: rebuild the network, put the trained weights in, and keep the calibration/threshold (in ck)
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
    """image: path or PIL image. Probabilities are calibrated (Platt scaling fitted on the val sets in train.py)."""
    ck, nets, device = model or load_model()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    z = ensemble_logits(nets, eval_tf(ck["img_size"])(img.convert("RGB")).unsqueeze(0).to(device))[0].cpu().numpy()
    # calibration: probability = sigmoid(a * score + b), then flagged if probability >= threshold
    p = 1 / (1 + np.exp(-(np.array(ck["calib_a"]) * z + np.array(ck["calib_b"]))))
    return {"conditions": [dict(name=c, probability=float(pi), threshold=float(t), flagged=bool(pi >= t))
                           for c, pi, t in zip(ck["classes"], p, ck["thresholds"])]}


if __name__ == "__main__":
    import sys

    print(predict(Path(sys.argv[1])))
