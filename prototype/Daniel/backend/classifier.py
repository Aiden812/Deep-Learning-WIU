"""Oral-condition classifier with a clearly labelled demo fallback.

If torch/torchvision are installed and the checkpoint exists, this runs the same
inference contract as models/classifier/demo/Daniel/app.py:
EXIF-orient -> RGB -> thumbnail <= cache side -> resize -> ImageNet normalise ->
logits / temperature -> softmax.

Otherwise it runs in demo mode: probabilities are simulated (deterministically from
the image bytes, or forced with a `scenario`) and every response says so.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO / "models" / "classifier" / "weights" / "Daniel" / "best.pt"
WEIGHTS = Path(os.environ.get("GIGI_WEIGHTS", DEFAULT_WEIGHTS))

# Order and IDs match shared/conditions.yaml.
CLASSES = ["calculus", "caries", "hypodontia", "mouth_ulcer", "normal", "tooth_discoloration"]
DEFAULT_ABSTAIN = 0.5

_KNOWN_HEADS = {"convnext": "classifier.2", "efficientnet": "classifier.1", "resnet": "fc", "regnet": "fc",
                "mobilenet_v3": "classifier.3", "swin": "head", "vit": "heads.head"}


class Classifier:
    def __init__(self) -> None:
        self.mode = "demo"
        self.reason = ""
        self.classes = list(CLASSES)
        self.abstain_threshold = DEFAULT_ABSTAIN
        self.backbone = None
        self._model = None
        try:
            self._load()
            self.mode = "model"
        except Exception as exc:  # torch missing, weights missing, or incompatible checkpoint
            self.reason = str(exc)

    # ---- real model -------------------------------------------------------
    def _load(self) -> None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"weights not found at {WEIGHTS}")
        import torch
        import torch.nn as nn
        import torchvision as tv
        from torchvision.transforms import v2

        try:
            ckpt = torch.load(WEIGHTS, map_location="cpu")
        except Exception:
            ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

        name = ckpt.get("backbone", "convnext_tiny")
        classes = list(ckpt["classes"])
        head = ckpt.get("head") or next((h for p, h in _KNOWN_HEADS.items() if name.startswith(p)), None)
        if head is None:
            raise ValueError(f"unknown backbone family {name!r}")
        m = tv.models.get_model(name, weights=None)
        parent_path, _, attr = head.rpartition(".")
        parent = m.get_submodule(parent_path) if parent_path else m
        old = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
        new = nn.Linear(old.in_features, len(classes))
        if attr.isdigit():
            parent[int(attr)] = new
        else:
            setattr(parent, attr, new)
        m.load_state_dict(ckpt["state_dict"])

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = m.to(self._device).eval()
        size = int(ckpt.get("img_size", 224))
        self._tf = v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(tuple(ckpt.get("mean", (0.485, 0.456, 0.406))), tuple(ckpt.get("std", (0.229, 0.224, 0.225)))),
        ])
        self._temperature = float(ckpt.get("temperature", 1.0))
        self._cache_side = int((ckpt.get("config") or {}).get("cache_max_side", 384))
        self.classes = classes
        self.abstain_threshold = float(ckpt.get("abstain_threshold", DEFAULT_ABSTAIN))
        self.backbone = name

    def _predict_model(self, im: Image.Image) -> dict[str, float]:
        torch = self._torch
        im = im.copy()
        im.thumbnail((self._cache_side, self._cache_side), Image.LANCZOS)
        with torch.no_grad():
            x = self._tf(im)[None].to(self._device)
            logits = self._model(x).float() / self._temperature
            probs = logits.softmax(1)[0].cpu().tolist()
        return {c: float(p) for c, p in zip(self.classes, probs)}

    # ---- demo fallback ----------------------------------------------------
    def _predict_demo(self, data: bytes, scenario: str | None) -> dict[str, float]:
        digest = hashlib.sha256(data).digest()
        if scenario in self.classes:
            top, conf = scenario, 0.86
        elif scenario == "unsure":
            top, conf = self.classes[digest[0] % len(self.classes)], 0.41
        else:
            top = self.classes[digest[0] % len(self.classes)]
            conf = 0.44 + (digest[1] / 255) * 0.5  # 0.44 to 0.94, so abstention shows up sometimes
        others = [c for c in self.classes if c != top]
        weights = [1 + digest[2 + i] for i in range(len(others))]
        rest = 1 - conf
        probs = {top: conf}
        for c, w in zip(others, weights):
            probs[c] = rest * w / sum(weights)
        return probs

    # ---- public -----------------------------------------------------------
    def predict(self, im: Image.Image, data: bytes, scenario: str | None = None) -> dict:
        t0 = time.perf_counter()
        probs = self._predict_model(im) if self.mode == "model" else self._predict_demo(data, scenario)
        ms = (time.perf_counter() - t0) * 1000
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top, conf = ranked[0]
        return {
            "mode": self.mode,
            "top": top,
            "confidence": conf,
            "abstain": conf < self.abstain_threshold,
            "abstain_threshold": self.abstain_threshold,
            "ranked": [{"name": n, "prob": p} for n, p in ranked],
            "inference_ms": round(ms, 1),
        }


def open_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    if im.format == "JPEG":
        im.draft("RGB", (1024, 1024))
    return ImageOps.exif_transpose(im).convert("RGB")
