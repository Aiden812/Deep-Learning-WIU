"""Gigi: oral-health screening prototype (backend + static frontend).

Run:   python prototype/Daniel/backend/app.py      then open http://127.0.0.1:8720

Endpoints
  GET  /api/meta          model mode, classes, test metrics, content
  POST /api/screen        multipart `file` (+ optional `scenario` in demo mode)
  POST /api/estimate      {condition, residency, tier, age}
  GET  /api/areas         towns for the clinic search
  POST /api/agent         {condition, abstain, profile, area | lat+lng, after_hours}

Privacy: photos are read into memory, screened and discarded. Nothing is written to
disk or logged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
import costs  # noqa: E402
import quality  # noqa: E402
from classifier import Classifier, open_image  # noqa: E402

FRONTEND = HERE.parent / "frontend"
METRICS = HERE.parents[2] / "models" / "classifier" / "results" / "Daniel" / "test_metrics.json"
MAX_MB = 15

CLF = Classifier()
CONTENT = json.loads((HERE / "data" / "conditions.json").read_text(encoding="utf-8"))
AREAS = json.loads((HERE / "data" / "areas.json").read_text(encoding="utf-8"))["areas"]

app = FastAPI(title="Gigi API", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _metrics() -> dict | None:
    if not METRICS.exists():
        return None
    try:
        m = json.loads(METRICS.read_text(encoding="utf-8"))
    except Exception:
        return None
    t = m.get("test", {})
    ext = m.get("external_test", {})
    return {
        "experiment": m.get("experiment"),
        "n_test": m.get("n_test"),
        "accuracy": t.get("acc"),
        "balanced_accuracy": t.get("bal_acc"),
        "macro_f1": t.get("macro_f1"),
        "macro_f1_ci": m.get("test_macro_f1_95ci"),
        "recall": {c: t.get(f"recall_{c}") for c in CLF.classes},
        "f1": {c: t.get(f"f1_{c}") for c in CLF.classes},
        "external_caries_recall": ext.get("exttest_caries_recall"),
        "robustness": m.get("robustness", []),
    }


@app.get("/api/meta")
def meta():
    return {
        "mode": CLF.mode,
        "mode_reason": CLF.reason if CLF.mode == "demo" else None,
        "backbone": CLF.backbone,
        "abstain_threshold": CLF.abstain_threshold,
        "classes": CLF.classes,
        "content": CONTENT,
        "metrics": _metrics(),
    }


@app.get("/api/areas")
def areas():
    return [a["name"] for a in AREAS]


@app.post("/api/screen")
async def screen(file: UploadFile = File(...), scenario: str | None = Form(None)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "The file was empty.")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"That photo is larger than {MAX_MB} MB. Try a smaller one.")
    try:
        im = open_image(data)
    except Exception:
        raise HTTPException(400, "That file couldn't be opened as a photo. Try a JPG or PNG.")

    q = quality.check(im)
    result = CLF.predict(im, data, scenario if CLF.mode == "demo" else None) if q["ok"] else None
    del data, im
    return {"quality": q, "result": result}


class EstimateIn(BaseModel):
    condition: str
    residency: str = "citizen"
    tier: str = "none"
    age: str = "under60"


@app.post("/api/estimate")
def estimate(body: EstimateIn):
    try:
        return costs.estimate(body.condition, body.residency, body.tier, body.age)
    except ValueError as e:
        raise HTTPException(400, str(e))


class AgentIn(BaseModel):
    condition: str
    abstain: bool = False
    profile: dict = {}
    area: str | None = None
    lat: float | None = None
    lng: float | None = None
    after_hours: bool = False


@app.post("/api/agent")
def run_agent(body: AgentIn):
    try:
        return agent.run(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GIGI_PORT", "8720"))
    print(f"classifier: {CLF.mode}" + (f" ({CLF.reason})" if CLF.mode == "demo" else f" ({CLF.backbone})"))
    print(f"Gigi: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
