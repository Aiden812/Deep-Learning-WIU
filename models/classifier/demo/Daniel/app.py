"""Web demo for the oral-condition classifier checkpoint.

Serves a single-page uploader on top of models/classifier/weights/Daniel/binary/best.pt
— the Stage-1 issue gate (override the file with DEMO_WEIGHTS, e.g. point it at the
six-class models/classifier/weights/Daniel/best.pt). The checkpoint is self-describing
and the demo adapts to its `task`:
  - "binary" (the default file): p(issue) at/above `threshold` flags the photo for a
    dental check.
  - "multiclass" (default task when absent): the 6-condition classifier; a top-class
    probability below `abstain_threshold` is flagged as an abstention.
Inference follows the checkpoint's own metadata (backbone, head, classes, mean/std,
temperature, threshold), the same preprocessing as the notebook's predict_image():
EXIF-orient -> RGB -> thumbnail <=384px -> squash-resize to img_size -> ImageNet
normalise -> logits / temperature -> softmax.

Anomalies (flagged by the gate, or a non-"normal" top class) are then localised by the
YOLO detector from models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb
(detector.py; DEMO_DETECTOR=0 disables it).

Run:  python models/classifier/demo/Daniel/app.py   (or run.bat)
Then open the printed URL in a browser.
"""

from __future__ import annotations

import base64
import io
import json
import os
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
WEIGHTS = Path(os.environ.get("DEMO_WEIGHTS") or
               REPO / "models" / "classifier" / "weights" / "Daniel" / "binary" / "best.pt")
# results/Daniel/{test_metrics.json,binary/test_metrics.json} — set below once the task is known.
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
NOTES = ckpt.get("notes", "")
CACHE_MAX_SIDE = int((ckpt.get("config") or {}).get("cache_max_side", 384))

# Two checkpoint schemas come out of the notebooks:
#   multiclass — 6 conditions, stores "abstain_threshold" (answer only above it);
#   binary     — Stage-1 normal-vs-issue gate, stores "threshold" (p(issue) cut)
#                and "positive_class". Both keep working through one code path.
TASK = str(ckpt.get("task", "multiclass"))
POSITIVE = ckpt.get("positive_class")
POS_IDX = CLASSES.index(POSITIVE) if POSITIVE in CLASSES else len(CLASSES) - 1
if TASK == "binary":
    ABSTAIN_THRESHOLD = float(ckpt.get("threshold", 0.5))  # decision cut on p(issue)
else:
    ABSTAIN_THRESHOLD = float(ckpt.get("abstain_threshold", 0.0))
TARGET_SENSITIVITY = ckpt.get("target_sensitivity")

# The matching test report lives next to the task's checkpoints (results/Daniel/binary/...).
METRICS = Path(os.environ["DEMO_METRICS"]) if os.environ.get("DEMO_METRICS") else \
    REPO / "models" / "classifier" / "results" / "Daniel" / ("binary" if TASK == "binary" else "") / "test_metrics.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        # parse_constant maps NaN/Infinity (the binary report writes them for
        # single-class per-source slices) to null so browsers can parse the JSON.
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    except Exception:
        return None


def _load_metrics() -> dict | None:
    return _load_json(METRICS)


def _detector_metrics() -> dict | None:
    # Just the test-set numbers the page shows (the full report also holds the training config).
    d = _load_json(DETECTOR_METRICS)
    if not d:
        return None
    t = d.get("test_all") or {}
    return {"mAP50": t.get("mAP50"), "mAP50_95": t.get("mAP50_95"),
            "per_class_AP50": {k: v.get("AP50") for k, v in (t.get("per_class") or {}).items()},
            "screening": ((d.get("image_level") or {}).get("screening") or {}).get("all test images")}

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

# Mouth gate (mouth_gate.py): routes each upload as an intraoral close-up (classified
# as-is), a face photo (mouth located and cropped first) or no usable mouth (rejected
# before the classifier runs). DEMO_MOUTH_GATE=0 turns it off; if it can't load
# (e.g. offline on first run, before CLIP is cached) the demo classifies whole images.
GATE = None
GATE_ERROR = None
if os.environ.get("DEMO_MOUTH_GATE", "1") != "0":
    try:
        from mouth_gate import CLIP_NAME, MouthGate
        GATE = MouthGate(DEVICE, HERE / "assets")
        GATE(Image.new("RGB", (320, 240), (128, 128, 128)))  # warm-up, like _warmup()
    except Exception as e:  # noqa: BLE001 — the demo must still start without the gate
        GATE_ERROR = f"{type(e).__name__}: {e}"
        print(f"mouth gate disabled ({GATE_ERROR})")
# Stage-2 detector (detector.py): when the photo is flagged (binary gate) or the top class
# isn't "normal" (6-class checkpoint), the YOLO model from yolo_oral_detector.ipynb boxes
# the conditions. DEMO_DETECTOR=0 turns it off, or points at another weights file.
DETECTOR_WEIGHTS = REPO / "models" / "yolo_caries" / "weights" / "Daniel" / "best.pt"
DETECTOR = None
DETECTOR_ERROR = None
if os.environ.get("DEMO_DETECTOR", "1") != "0":
    try:
        from detector import ConditionDetector
        _dw = os.environ.get("DEMO_DETECTOR", "1")
        DETECTOR = ConditionDetector(Path(_dw) if _dw != "1" else DETECTOR_WEIGHTS, DEVICE)
    except Exception as e:  # noqa: BLE001 — the classifier demo must still start without it
        DETECTOR_ERROR = f"{type(e).__name__}: {e}"
        print(f"condition detector disabled ({DETECTOR_ERROR})")
DETECTOR_METRICS = REPO / "models" / "yolo_caries" / "results" / "Daniel" / "test_metrics.json"
NORMAL_CLASSES = {"normal"}

# Decode large enough that a mouth cropped out of a face photo still has detail
# (and the detector, which runs at 640 px, gets a full-resolution input).
WORK_SIDE = 1280 if (GATE or DETECTOR) else CACHE_MAX_SIDE


def _jpeg_data_url(im: Image.Image, side: int = 320) -> str:
    im = im.copy()
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def predict_bytes(data: bytes) -> dict:
    try:
        im = Image.open(io.BytesIO(data))
        if im.format == "JPEG":
            im.draft("RGB", (WORK_SIDE, WORK_SIDE))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE), Image.LANCZOS)
    except Exception:
        raise HTTPException(400, "Could not decode that file as an image.")

    gate = None
    if GATE is not None:
        t0 = time.perf_counter()
        g = GATE(im)
        gate = g.as_dict() | {"gate_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        if g.route == "rejected":
            return {"rejected": True, "gate": gate}
        im = g.image
        if g.route == "face_crop":
            gate["crop_image"] = _jpeg_data_url(im)
    full = im  # what the detector sees, before the classifier's thumbnail
    im = im.copy()
    im.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.LANCZOS)
    x = _TF(im)[None].to(DEVICE)
    t0 = time.perf_counter()
    logits = MODEL(x).float() / TEMPERATURE
    probs = logits.softmax(1)[0].float().cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000.0

    order = probs.argsort()[::-1]
    k = int(order[0])
    out = {
        "class_id": k,
        "class_name": CLASSES[k],
        "display_name": DISPLAY.get(CLASSES[k], CLASSES[k]),
        "confidence": float(probs[k]),
        "top2": [{"class": CLASSES[i], "display_name": DISPLAY.get(CLASSES[i], CLASSES[i]),
                  "prob": float(probs[i])} for i in order[:2]],
        "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "inference_ms": round(ms, 1),
        "rejected": False,
        "gate": gate,
    }
    if TASK == "binary":
        # Verdict = p(issue) vs the tuned decision threshold, not argmax: the band
        # between 0.5 and the cut is argmax-issue but deliberately left unflagged.
        p_issue = float(probs[POS_IDX])
        flagged = p_issue >= ABSTAIN_THRESHOLD
        k = POS_IDX if flagged else 1 - POS_IDX
        out.update({"class_id": k, "class_name": CLASSES[k], "display_name": DISPLAY.get(CLASSES[k], CLASSES[k]),
                    "confidence": float(probs[k]),
                    "task": "binary", "p_issue": p_issue, "flagged": flagged,
                    "threshold": ABSTAIN_THRESHOLD, "abstain": flagged})
    else:
        out["abstain"] = bool(probs[k] < ABSTAIN_THRESHOLD)

    # Anomaly => localise it with the detector.
    anomaly = out["flagged"] if TASK == "binary" else CLASSES[k].lower() not in NORMAL_CLASSES
    out["anomaly"] = bool(anomaly)
    out["detection"] = None
    if anomaly and DETECTOR is not None:
        t0 = time.perf_counter()
        boxes = DETECTOR(full, always=3)  # flagged => always outline something
        out["detection"] = {
            "boxes": boxes,
            "n_confident": sum(b["confident"] for b in boxes),
            "conf": DETECTOR.conf,
            "detector_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "image": _jpeg_data_url(full, 720),  # boxes are fractions of this image
        }
    return out


app = FastAPI(title="Oral condition classifier demo", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/meta")
def meta():
    out = {
        "weights": str(WEIGHTS),
        "backbone": BACKBONE,
        "task": TASK,
        "experiment": ckpt.get("experiment"),
        "classes": [{"id": i, "name": c, "display_name": DISPLAY.get(c, c)} for i, c in enumerate(CLASSES)],
        "temperature": TEMPERATURE,
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "created": ckpt.get("created"),
        "notes": NOTES,
        "metrics": _load_metrics(),
        "mouth_gate": {"enabled": GATE is not None, "clip": CLIP_NAME if GATE else None,
                       "error": GATE_ERROR},
        "detector": DETECTOR.info() | {"metrics": _detector_metrics()} if DETECTOR
        else {"enabled": False, "error": DETECTOR_ERROR},
    }
    if TASK == "binary":
        out["decision_threshold"] = ABSTAIN_THRESHOLD
        out["positive_class"] = POSITIVE
        out["target_sensitivity"] = TARGET_SENSITIVITY
    return out


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
    import uvicorn

    port = int(os.environ.get("DEMO_PORT", "8712"))
    if TASK == "binary":
        print(f"model: {BACKBONE} binary gate on {DEVICE} | "
              f"T={TEMPERATURE:.3f} flag p(issue)>={ABSTAIN_THRESHOLD:.2f}")
    else:
        print(f"model: {BACKBONE} ({len(CLASSES)} classes) on {DEVICE} | "
              f"T={TEMPERATURE:.3f} abstain<{ABSTAIN_THRESHOLD:.2f}")
    print(f"weights: {WEIGHTS}")
    print("mouth gate: " + (f"on (YuNet + {CLIP_NAME})" if GATE else f"off ({GATE_ERROR or 'DEMO_MOUTH_GATE=0'})"))
    print("detector: " + (f"on ({DETECTOR.run}, conf>={DETECTOR.conf:.2f}) {DETECTOR.weights}" if DETECTOR
                          else f"off ({DETECTOR_ERROR or 'DEMO_DETECTOR=0'})"))
    print(f"demo: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
