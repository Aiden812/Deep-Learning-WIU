"""Web demo for the oral-condition classifier checkpoint.

Serves a single-page uploader on top of models/classifier/weights/Daniel/best.pt.
Same inference contract as the notebook's predict_image() / src/predict.py:
EXIF-orient -> RGB -> thumbnail <=384px -> squash-resize to img_size -> ImageNet
normalise -> logits / temperature -> softmax.

Run:  python models/classifier/demo/Daniel/app.py   (or run.bat)
Then open the printed URL in a browser.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # .../Deep-Learning-WIU
WEIGHTS = REPO / "models" / "classifier" / "weights" / "Daniel" / "best.pt"
METRICS = REPO / "models" / "classifier" / "results" / "Daniel" / "test_metrics.json"

MAX_UPLOAD_MB = 20


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Weights not found at {path}. Expected the training run to have saved best.pt there.")
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


# Final-layer path per backbone, for checkpoints saved before the notebook stored "head" itself.
_KNOWN_HEADS = {"convnext": "classifier.2", "efficientnet": "classifier.1", "resnet": "fc", "regnet": "fc",
                "mobilenet_v3": "classifier.3", "densenet": "classifier", "swin": "head", "vit": "heads.head"}


def _build_model(name: str, n: int, head: str | None = None) -> nn.Module:
    # Mirrors build_model() in oral_condition_classifier.ipynb: any torchvision classifier with its
    # final nn.Linear (at attribute path `head`) replaced by an n-way layer.
    if head is None:
        head = next((h for prefix, h in _KNOWN_HEADS.items() if name.startswith(prefix)), None)
        if head is None:
            raise ValueError(f"Checkpoint has no 'head' and backbone {name!r} is not a known family")
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


ckpt = _load_checkpoint(WEIGHTS)
CLASSES = list(ckpt["classes"])
DISPLAY = dict(ckpt.get("display_names") or {c: c.replace("_", " ").title() for c in CLASSES})
BACKBONE = ckpt.get("backbone", "convnext_tiny")
IMG_SIZE = int(ckpt.get("img_size", 224))
MEAN = tuple(ckpt.get("mean", (0.485, 0.456, 0.406)))
STD = tuple(ckpt.get("std", (0.229, 0.224, 0.225)))
TEMPERATURE = float(ckpt.get("temperature", 1.0))
ABSTAIN_THRESHOLD = float(ckpt.get("abstain_threshold", 0.0))
NOTES = ckpt.get("notes", "")
CACHE_MAX_SIDE = int((ckpt.get("config") or {}).get("cache_max_side", 384))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = _build_model(BACKBONE, len(CLASSES), ckpt.get("head"))
MODEL.load_state_dict(ckpt["state_dict"])
MODEL.to(DEVICE).eval()

_TF = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])


def _warmup():
    # Absorb CUDA/kernel init cost before the first real request.
    dummy = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
    for _ in range(3):
        with torch.no_grad():
            MODEL(dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


_warmup()


@torch.no_grad()
def predict_bytes(data: bytes) -> dict:
    try:
        im = Image.open(io.BytesIO(data))
        if im.format == "JPEG":
            im.draft("RGB", (CACHE_MAX_SIDE, CACHE_MAX_SIDE))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.LANCZOS)
    except Exception:
        raise HTTPException(400, "Could not decode that file as an image.")
    x = _TF(im)[None].to(DEVICE)
    t0 = time.perf_counter()
    logits = MODEL(x).float() / TEMPERATURE
    probs = logits.softmax(1)[0].float().cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000.0

    order = probs.argsort()[::-1]
    k = int(order[0])
    return {
        "class_id": k,
        "class_name": CLASSES[k],
        "display_name": DISPLAY.get(CLASSES[k], CLASSES[k]),
        "confidence": float(probs[k]),
        "abstain": bool(probs[k] < ABSTAIN_THRESHOLD),
        "top2": [{"class": CLASSES[i], "display_name": DISPLAY.get(CLASSES[i], CLASSES[i]),
                  "prob": float(probs[i])} for i in order[:2]],
        "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "inference_ms": round(ms, 1),
    }


app = FastAPI(title="Oral condition classifier demo", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/meta")
def meta():
    metrics = None
    if METRICS.exists():
        try:
            metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        except Exception:
            metrics = None
    return {
        "weights": str(WEIGHTS),
        "backbone": BACKBONE,
        "experiment": ckpt.get("experiment"),
        "classes": [{"id": i, "name": c, "display_name": DISPLAY.get(c, c)} for i, c in enumerate(CLASSES)],
        "temperature": TEMPERATURE,
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "created": ckpt.get("created"),
        "notes": NOTES,
        "metrics": metrics,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(400, "Empty upload.")
    return predict_bytes(data)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("DEMO_PORT", "8712"))
    print(f"model: {BACKBONE} ({len(CLASSES)} classes) on {DEVICE} | "
          f"T={TEMPERATURE:.3f} abstain<{ABSTAIN_THRESHOLD:.2f}")
    print(f"demo: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
