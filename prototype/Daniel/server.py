"""SmileCheck SG backend: serves this prototype page on top of Daniel's models.

The screening pipeline is the one in models/classifier/demo/Daniel/app.py, imported rather
than copied so both front ends always run the same models, thresholds and preprocessing:

  1. mouth gate (mouth_gate.py)   - intraoral close-up / face photo cropped to the mouth / rejected
  2. issue gate (binary best.pt)  - calibrated p(issue), flagged at the checkpoint's threshold
  3. YOLO detector (detector.py)  - boxes Calculus, Caries, Hypodontia, Mouth_Ulcer,
                                    Tooth_Discoloration on flagged photos

The demo's environment overrides (DEMO_WEIGHTS, DEMO_MOUTH_GATE, DEMO_DETECTOR, ...) apply
here too; pointing DEMO_WEIGHTS at the six-class checkpoint also works.

Step 2 can be switched off (PROTO_CLASSIFIER=0, or the form field `classifier=0` per request):
YOLO then runs on every accepted photo and frame, and a photo is flagged when it has a box at
or above the detector's own operating point (best.json `conf`). No forced best guesses are drawn.

The page sends any number of photos and videos. Every photo, and every frame of every video,
goes through the pipeline on its own; the results are then combined per source and across
sources (see _combine). Progress is streamed back as NDJSON so long videos show a counter.

Run from the repo root:  python prototype/Daniel/server.py   (or run.bat)
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEMO_DIR = REPO / "models" / "classifier" / "demo" / "Daniel"

# The demo imports mouth_gate / detector as top-level modules from its own folder.
sys.path.insert(0, str(DEMO_DIR))
_spec = importlib.util.spec_from_file_location("daniel_demo", DEMO_DIR / "app.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)  # loads the classifier, mouth gate and detector once

MAX_UPLOAD_MB = demo.MAX_UPLOAD_MB          # per photo
MAX_VIDEO_MB = int(os.environ.get("VIDEO_MAX_MB", "300"))
# Every frame is checked. Longer videos are sampled evenly down to this many frames.
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "1800"))
# A video is flagged when at least this share of its readable frames is flagged (and at
# least VIDEO_MIN_FLAGGED frames), so a single blurry frame can't flag a whole video.
VIDEO_FLAG_FRACTION = float(os.environ.get("VIDEO_FLAG_FRACTION", "0.1"))
VIDEO_MIN_FLAGGED = 2
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv", ".3gp"}

# Issue classifier on (default) or off. Off = detector-only screening, which needs the detector.
# The classifier is still loaded by the demo module; it just isn't run.
_OFF = {"0", "off", "false", "no"}
CLASSIFIER_DEFAULT = os.environ.get("PROTO_CLASSIFIER", "1").strip().lower() not in _OFF
if not CLASSIFIER_DEFAULT and demo.DETECTOR is None:
    raise SystemExit(f"PROTO_CLASSIFIER=0 needs the YOLO detector, which didn't load "
                     f"({demo.DETECTOR_ERROR or 'DEMO_DETECTOR=0'}).")


def _is_video(f: UploadFile) -> bool:
    return (f.content_type or "").startswith("video/") or Path(f.filename or "").suffix.lower() in VIDEO_EXT


def _decode(data: bytes) -> Image.Image:
    """Same decoding as demo.predict_bytes()."""
    try:
        im = Image.open(io.BytesIO(data))
        if im.format == "JPEG":
            im.draft("RGB", (demo.WORK_SIDE, demo.WORK_SIDE))
        return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode that file as an image.")


def _detect_only(im: Image.Image) -> dict:
    """Mouth gate -> YOLO, no classifier. Flagged = a box at/above the detector's `conf`;
    the item's score (kept in `p_issue` so _combine treats both modes alike) is its strongest
    box confidence, 0 if YOLO returns nothing at/above its weak cut."""
    im.thumbnail((demo.WORK_SIDE, demo.WORK_SIDE), Image.LANCZOS)
    gate = None
    if demo.GATE is not None:
        t0 = time.perf_counter()
        g = demo.GATE(im)
        gate = g.as_dict() | {"gate_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        if g.route == "rejected":
            return {"rejected": True, "gate": gate}
        im = g.image
    t0 = time.perf_counter()
    boxes = demo.DETECTOR(im)  # no `always`: nothing is forced onto a photo YOLO finds clean
    det = {"boxes": boxes, "n_confident": sum(b["confident"] for b in boxes), "conf": demo.DETECTOR.conf,
           "detector_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    if boxes:  # only photos with something to draw carry an image (keeps long videos light)
        det["image"] = demo._jpeg_data_url(im, 720)
    flagged = det["n_confident"] > 0
    return {"rejected": False, "gate": gate, "task": "binary", "classifier": False,
            "p_issue": max((b["conf"] for b in boxes), default=0.0), "flagged": flagged,
            "threshold": demo.DETECTOR.conf, "anomaly": flagged, "detection": det}


def _run(src: bytes | Image.Image, use_classifier: bool = True) -> dict:
    """One photo or frame through the pipeline; undecodable input counts as rejected."""
    try:
        if use_classifier:
            r = demo.predict_bytes(src) if isinstance(src, bytes) else demo.predict_image(src)
        else:
            r = _detect_only(_decode(src) if isinstance(src, bytes) else src)
    except HTTPException as e:
        return {"rejected": True, "gate": {"route": "rejected", "reason": "unreadable", "message": e.detail}}
    if r.get("gate"):
        r["gate"].pop("crop_image", None)  # the page doesn't use it
    return r


def _frame_plan(path: str) -> tuple[int | None, set[int] | None]:
    """(frames we will check, frame indices to keep when sampling) for a video file."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0, None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if n <= 0:
        return None, None  # unknown length (some webm files): read until the end or the cap
    if n <= VIDEO_MAX_FRAMES:
        return n, None
    return VIDEO_MAX_FRAMES, set(np.linspace(0, n - 1, VIDEO_MAX_FRAMES).round().astype(int).tolist())


def _frames(path: str, keep: set[int] | None):
    """Yields (frame index, seconds, RGB PIL image). OpenCV applies the rotation metadata."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    i = yielded = 0
    try:
        while yielded < VIDEO_MAX_FRAMES:
            if keep is not None and i not in keep:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            yield i, (round(i / fps, 2) if fps else None), Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            i += 1
            yielded += 1
    finally:
        cap.release()


def _box_key(item: dict):
    """(confident, conf) of an item's strongest detector box, or None."""
    boxes = (item.get("detection") or {}).get("boxes") or []
    return max(((b["confident"], b["conf"]) for b in boxes), default=None)


def _combine(items: list[dict], sources: list[dict], use_classifier: bool = True) -> dict:
    ok = [p for p in items if not p["rejected"]]
    # Detector-only screening scores each item by its strongest box, cut at the detector's conf.
    task = demo.TASK if use_classifier else "binary"
    threshold = demo.ABSTAIN_THRESHOLD if use_classifier else demo.DETECTOR.conf
    by_src: dict[int, list[dict]] = {}
    for p in ok:
        by_src.setdefault(p["source"], []).append(p)
    for s in sources:
        s["accepted"] = len(by_src.get(s["index"], []))
        s["reasons"] = dict(Counter((p.get("gate") or {}).get("reason") or "unreadable"
                                    for p in items if p["source"] == s["index"] and p["rejected"]))

    out = {"task": task, "classifier": use_classifier, "threshold": threshold, "photos": items,
           "sources": sources, "n_accepted": len(ok), "rejected": not ok}
    if not ok:
        return out

    if task == "binary":
        # Per source: a photo's own p(issue); for a video, the k-th highest frame p(issue), so the
        # video is flagged exactly when at least k frames are. Any flagged source flags the screening
        # (an issue may only show from one angle) and the ring shows the strongest source score.
        for s in sources:
            ps = sorted((p["p_issue"] for p in by_src.get(s["index"], [])), reverse=True)
            if not ps:
                s["flagged"] = False
                continue
            k = 1 if s["kind"] == "image" else min(len(ps), max(VIDEO_MIN_FLAGGED, math.ceil(VIDEO_FLAG_FRACTION * len(ps))))
            s["p_issue"] = ps[k - 1]
            s["flagged"] = s["p_issue"] >= threshold
            s["flagged_frames"] = sum(p["flagged"] for p in by_src[s["index"]])
        scored = [s for s in sources if "p_issue" in s]
        out["p_issue"] = max(s["p_issue"] for s in scored)
        out["flagged"] = any(s["flagged"] for s in scored)
        out["anomaly"] = out["flagged"]
        pool = [p for p in ok if sources[p["source"]]["flagged"]]
    else:
        # Six-class checkpoint: average within each source, then across sources, so a long video
        # doesn't outvote the photos.
        src_probs = [{c: sum(p["probs"][c] for p in ps) / len(ps) for c in demo.CLASSES} for ps in by_src.values()]
        probs = {c: sum(sp[c] for sp in src_probs) / len(src_probs) for c in demo.CLASSES}
        top = max(probs, key=probs.get)
        out.update(probs=probs, class_name=top, confidence=probs[top],
                   abstain=probs[top] < demo.ABSTAIN_THRESHOLD, anomaly=top.lower() not in demo.NORMAL_CLASSES)
        pool = ok

    # Pool the detector's boxes per condition across the photos and frames that count.
    findings: dict[str, dict] = {}
    best = None  # (box key, item index) of the strongest box, to pick the photo to show
    for p in pool:
        for b in (p.get("detection") or {}).get("boxes") or []:
            f = findings.setdefault(b["class"], {"class": b["class"], "display_name": b["display_name"],
                                                 "conf": 0.0, "n": 0, "confident": False, "sources": []})
            f["conf"] = max(f["conf"], b["conf"])
            f["n"] += 1
            f["confident"] = f["confident"] or b["confident"]
            if p["source"] not in f["sources"]:
                f["sources"].append(p["source"])
        key = _box_key(p)
        if key is not None and (best is None or key > best[0]):
            best = (key, p["index"])
    out["findings"] = sorted(findings.values(), key=lambda f: (not f["confident"], -f["conf"]))
    out["detector_ran"] = any(p.get("detection") for p in pool)
    out["detector_conf"] = demo.DETECTOR.conf if demo.DETECTOR else None
    out["show_photo"] = best[1] if best else (pool or ok)[0]["index"]

    # Only the shown item keeps its images, so a long video doesn't send hundreds of JPEGs.
    for p in items:
        if p["index"] != out["show_photo"]:
            p.pop("image", None)
            if p.get("detection"):
                p["detection"].pop("image", None)
    return out


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _stream(uploads: list[dict], use_classifier: bool = True):
    """Runs every photo and video frame, yielding NDJSON progress lines and then the result."""
    items: list[dict] = []
    sources: list[dict] = []
    try:
        for s, u in enumerate(uploads):
            src = {"index": s, "kind": u["kind"], "name": u["name"]}
            if u["kind"] == "video":
                src["planned"], u["keep"] = _frame_plan(u["path"])
            sources.append(src)
        planned = [1 if s["kind"] == "image" else s["planned"] for s in sources]
        total = None if None in planned else sum(planned)
        yield _line({"type": "start", "total": total})

        for s, u in zip(sources, uploads):
            if u["kind"] == "image":
                frames = [(None, None, u["data"])]
            else:
                frames = _frames(u["path"], u["keep"])
            first_ok = True
            # Per-source best box: only that frame's detector image is kept while running.
            src_best = None
            n = 0
            for frame, t, payload in frames:
                item = {"index": len(items), "source": s["index"], "frame": frame, "t": t, **_run(payload, use_classifier)}
                n += 1
                if u["kind"] == "video" and not item["rejected"]:
                    if first_ok:  # a still of the video, shown when nothing is outlined
                        item["image"] = demo._jpeg_data_url(payload, 720)
                        first_ok = False
                    key = _box_key(item)
                    if key is not None:
                        if src_best is None or key > src_best[0]:
                            if src_best:
                                src_best[1]["detection"].pop("image", None)
                            src_best = (key, item)
                        else:
                            item["detection"].pop("image", None)
                items.append(item)
                yield _line({"type": "progress", "done": len(items), "total": total})
            s["frames"] = n
            if u["kind"] == "video" and n == 0:  # OpenCV couldn't open or decode it
                items.append({"index": len(items), "source": s["index"], "frame": None, "t": None, "rejected": True,
                              "gate": {"route": "rejected", "reason": "unreadable_video",
                                       "message": "Could not read that video."}})

        yield _line({"type": "result", **_combine(items, sources, use_classifier)})
    except Exception as e:  # report instead of cutting the stream off mid-way
        yield _line({"type": "error", "detail": f"{type(e).__name__}: {e}"})
    finally:
        for u in uploads:
            if u.get("path"):
                try:
                    os.unlink(u["path"])
                except OSError:
                    pass


app = FastAPI(title="SmileCheck SG", docs_url="/api/docs", openapi_url="/api/openapi.json")
# Lets the page also be opened from a separate local static server during development.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["GET", "POST"])


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    return demo.meta() | {"classifier_enabled": CLASSIFIER_DEFAULT}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...), classifier: str | None = Form(None)):
    # `classifier` (optional form field: 1/on or 0/off) overrides PROTO_CLASSIFIER for this request.
    use_classifier = CLASSIFIER_DEFAULT if classifier is None else classifier.strip().lower() not in _OFF
    if not use_classifier and demo.DETECTOR is None:
        raise HTTPException(400, "The classifier can't be turned off: the YOLO detector isn't loaded.")
    if not files:
        raise HTTPException(400, "Send at least one photo or video.")
    uploads = []
    try:
        for f in files:
            data = await f.read()
            name = f.filename or "upload"
            if not data:
                raise HTTPException(400, f"{name} is empty.")
            if _is_video(f):
                if len(data) > MAX_VIDEO_MB * 1024 * 1024:
                    raise HTTPException(413, f"{name} is larger than {MAX_VIDEO_MB} MB.")
                # OpenCV only decodes from a path: a temp file, deleted as soon as the request ends.
                fd, path = tempfile.mkstemp(suffix=Path(name).suffix.lower() or ".mp4", prefix="smilecheck-")
                with os.fdopen(fd, "wb") as out:
                    out.write(data)
                uploads.append({"kind": "video", "name": name, "path": path})
            else:
                if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_MB} MB.")
                # Held in memory for this request only.
                uploads.append({"kind": "image", "name": name, "data": data})
    except HTTPException:
        for u in uploads:
            if u.get("path"):
                os.unlink(u["path"])
        raise
    return StreamingResponse(_stream(uploads, use_classifier), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


app.mount("/css", StaticFiles(directory=HERE / "css"), name="css")
app.mount("/js", StaticFiles(directory=HERE / "js"), name="js")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PROTO_PORT", "8713"))
    if not CLASSIFIER_DEFAULT:
        print(f"classifier: off (PROTO_CLASSIFIER=0) | detector only, flag box conf>={demo.DETECTOR.conf:.2f}")
    elif demo.TASK == "binary":
        print(f"classifier: {demo.BACKBONE} issue gate on {demo.DEVICE} | flag p(issue)>={demo.ABSTAIN_THRESHOLD:.2f}")
    else:
        print(f"classifier: {demo.BACKBONE} ({len(demo.CLASSES)} classes) on {demo.DEVICE}")
    print(f"weights: {demo.WEIGHTS}")
    print("mouth gate: " + ("on" if demo.GATE else f"off ({demo.GATE_ERROR or 'DEMO_MOUTH_GATE=0'})"))
    print("detector: " + (f"on ({demo.DETECTOR.run}, conf>={demo.DETECTOR.conf:.2f})" if demo.DETECTOR
                          else f"off ({demo.DETECTOR_ERROR or 'DEMO_DETECTOR=0'})"))
    print(f"videos: every frame, up to {VIDEO_MAX_FRAMES} per video")
    print(f"SmileCheck SG: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
