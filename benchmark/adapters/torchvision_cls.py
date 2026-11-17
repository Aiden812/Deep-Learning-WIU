"""Brandon's and Daniel's torchvision classifiers (self-describing checkpoints: backbone, head, classes,
temperature, threshold, config.cache_max_side). Same steps as their notebooks' screen()/predict_image():
EXIF-upright RGB -> thumbnail to cache_max_side (LANCZOS) -> squash-resize to img_size -> ImageNet normalise
-> softmax(logits / T)."""
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image
from torchvision.transforms import v2

from . import HEALTHY_NAMES, norm_name, to_ours

KNOWN_HEADS = {"convnext": "classifier.2", "efficientnet": "classifier.1", "resnet": "fc", "regnet": "fc",
               "mobilenet_v3": "classifier.3", "densenet": "classifier", "swin": "head", "vit": "heads.head"}


def build(name, n, head=None):
    head = head or next(h for p, h in KNOWN_HEADS.items() if name.startswith(p))
    m = tv.models.get_model(name, weights=None)
    parent_path, _, attr = head.rpartition(".")
    parent = m.get_submodule(parent_path) if parent_path else m
    old = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
    new = nn.Linear(old.in_features, n)
    if attr.isdigit():
        parent[int(attr)] = new
    else:
        setattr(parent, attr, new)
    return m


class TorchvisionCls:
    batchable = True

    def __init__(self, weights, **_):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.classes = list(ck["classes"])
        self.T = float(ck.get("temperature", 1.0))
        self.side = int((ck.get("config") or {}).get("cache_max_side", 384))
        self.task = ck.get("task", "multiclass")
        self.binary = self.task in ("issue_or_not", "binary") or len(self.classes) == 2
        pos = ck.get("positive_class")
        self.pos = self.classes.index(pos) if pos in self.classes else len(self.classes) - 1
        self.threshold = float(ck.get("threshold", 0.5)) if self.binary else None
        self.healthy = [i for i, c in enumerate(self.classes) if norm_name(c) in HEALTHY_NAMES]
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build(ck.get("backbone", "convnext_tiny"), len(self.classes), ck.get("head"))
        self.model.load_state_dict(ck["state_dict"])
        self.model.float().eval().to(self.dev)
        img = int(ck.get("img_size", 224))
        self.tf = v2.Compose([v2.ToImage(), v2.Resize((img, img), antialias=True), v2.ToDtype(torch.float32, scale=True),
                              v2.Normalize(ck.get("mean", (0.485, 0.456, 0.406)), ck.get("std", (0.229, 0.224, 0.225)))])

    def prep(self, im: Image.Image):
        im = im.copy()
        im.thumbnail((self.side, self.side), Image.LANCZOS)
        return self.tf(im)

    @torch.no_grad()
    def predict_batch(self, ims):
        x = torch.stack([self.prep(im) for im in ims]).to(self.dev)
        probs = (self.model(x).float() / self.T).softmax(1).cpu().numpy()
        out = []
        for pr in probs:
            if self.binary:
                p = float(pr[self.pos])
                out.append(dict(p_problem=p, flag_problem=p >= self.threshold, p={}, flag={}, boxes=None))
                continue
            k = int(pr.argmax())
            p_h = float(pr[self.healthy].sum()) if self.healthy else None
            per = {to_ours(c): float(pr[i]) for i, c in enumerate(self.classes) if to_ours(c)}
            out.append(dict(p_problem=None if p_h is None else 1 - p_h,
                            flag_problem=None if p_h is None else k not in self.healthy,
                            p=per, flag={to_ours(c): k == i for i, c in enumerate(self.classes) if to_ours(c)}, boxes=None))
        return out
