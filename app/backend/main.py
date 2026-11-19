"""SmileCheck SG backend: serves the prototype page and the condition classifier.

Run from the repo root:  python -m uvicorn app.backend.main:app --port 8000
"""

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype" / "Brandon" / "backend"))

import costs  # noqa: E402  (prototype/Brandon/backend/costs.py)
from genai.rag import answer as explainer  # noqa: E402
from models.classifier.src.predict import Classifier  # noqa: E402
from models.quality_check.src.predict import check as quality_check  # noqa: E402

MAX_PHOTOS = 5
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="SmileCheck SG")
# Lets the page also be opened from a separate static server during development.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
)

classifier = Classifier()


@app.get("/api/health")
def health():
    return {"ok": True}


async def _read_image(f: UploadFile) -> Image.Image:
    data = await f.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{f.filename} is too large.")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{f.filename} is not an image we can read.")
    return im


@app.post("/api/quality")
async def quality(file: UploadFile = File(...)):
    # Runs as soon as a photo is added, so the checklist can tick or cross each item. CPU only, ~50 ms.
    return quality_check(await _read_image(file))


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= MAX_PHOTOS:
        raise HTTPException(400, f"Send between 1 and {MAX_PHOTOS} photos.")
    # Photos are only held in memory for this request and never written to disk.
    images = [await _read_image(f) for f in files]
    # Quality gate (models/quality_check): photos that fail are left out, so the model never guesses on them.
    # Results come back in upload order, so the page can say which angle was skipped and why.
    checks = [quality_check(im) for im in images]
    usable = [im for im, q in zip(images, checks) if q["ok"]]
    if not usable:
        return {"rejected": True, "quality": checks}
    return {**classifier.predict_images(usable), "quality": checks}


class EstimateIn(BaseModel):
    condition: str
    residency: str = "citizen"
    tier: str = "unknown"
    age: str = "under60"


@app.post("/api/estimate")
def estimate(body: EstimateIn):
    # Only these three self-selected answers are used; nothing is stored.
    try:
        return costs.estimate(body.condition, body.residency, body.tier, body.age)
    except ValueError as e:
        raise HTTPException(400, str(e))


class AskIn(BaseModel):
    question: str
    condition: str | None = None
    residency: str = "citizen"
    tier: str = "unknown"
    age: str = "under60"


@app.post("/api/ask")
def ask(body: AskIn):
    # "Ask about your result" (genai/rag). The question is answered and discarded: nothing is logged.
    out = explainer.ask(body.question, body.condition)
    if out["show_cost"] and body.condition and out["route"] == "answer":
        try:  # their own estimate from the cost section, so the numbers match it exactly
            est = costs.estimate(body.condition, body.residency, body.tier, body.age)
            out["cost"] = {"pay_min": est["likely"]["total"]["pay_min"], "pay_max": est["likely"]["total"]["pay_max"],
                           "has_items": bool(est["likely"]["lines"])}
        except ValueError:
            pass
    return out


@app.get("/api/ask/suggestions")
def ask_suggestions(condition: str | None = None):
    return {"questions": explainer.suggestions(condition), "llm": explainer.llm.provider()}


app.mount("/", StaticFiles(directory=ROOT / "prototype", html=True), name="frontend")
