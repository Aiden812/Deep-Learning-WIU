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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.classifier.src.predict import Classifier  # noqa: E402

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


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= MAX_PHOTOS:
        raise HTTPException(400, f"Send between 1 and {MAX_PHOTOS} photos.")
    images = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename} is too large.")
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except UnidentifiedImageError:
            raise HTTPException(400, f"{f.filename} is not an image we can read.")
        images.append(im)
    # Photos are only held in memory for this request and never written to disk.
    return classifier.predict_images(images)


app.mount("/", StaticFiles(directory=ROOT / "prototype", html=True), name="frontend")
