"""Senyum: oral-health screening prototype (Aiden). Backend + static frontend.

Run:   python prototype/Aiden/backend/app.py      then open http://127.0.0.1:8730

  GET  /api/meta      model mode, plain-language content, towns, model test metrics
  POST /api/screen    multipart `file`, or `demo` = comma-separated condition ids for a simulated result
  POST /api/plan      {conditions, profile, area | lat+lng, after_hours} -> agent trace, cost, clinics, products

Unlike a single-label classifier, the model answers yes/no for each of five conditions, so one photo can come back with
several findings (e.g. tartar and discolouration); cost and clinic search then cover all of them at once.

Privacy: photos are read into memory, screened and discarded. Nothing is written to disk or logged.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# shortcut: cost, clinic, product, guardrail and photo-quality code is imported from Daniel's prototype, not copied;
# move it to cost/ and genai/agent/ when the team merges prototypes into app/.
DANIEL = REPO / "prototype" / "Daniel" / "backend"
sys.path[:0] = [str(DANIEL), str(REPO / "models" / "classifier" / "src")]

import agent as tools  # noqa: E402  (Daniel's: find_clinics, safety_review, PRODUCTS, AREAS)
import costs  # noqa: E402
import quality  # noqa: E402
from classifier import open_image  # noqa: E402

# which trained model to serve: a folder name under results/Aiden (densenet121 or dinov2_vitb14)
RESULTS = REPO / "models" / "classifier" / "results" / "Aiden" / os.environ.get("SENYUM_MODEL", "dinov2_vitb14")
WEIGHTS = RESULTS / "model.pt"
MAX_MB = 15
BORDERLINE = 0.15   # a probability this close below its threshold is shown as "borderline", not "clear"
RANK = {"routine": 0, "watch": 1, "soon": 2}

_content = json.loads((DANIEL / "data" / "conditions.json").read_text(encoding="utf-8"))
CONDITIONS = _content["conditions"]
ORDER = ["calculus", "caries", "hypodontia", "mouth_ulcer", "tooth_discoloration"]   # demo mode; a loaded model sets its own

MODEL, MODE_REASON = None, ""
try:
    import multilabel
    MODEL = multilabel.load_model(WEIGHTS)
    ORDER = [c.lower() for c in MODEL[0]["classes"]]
    assert set(ORDER) <= set(CONDITIONS), f"no plain-language content for {set(ORDER) - set(CONDITIONS)}"
except Exception as e:   # no torch, no weights, or an unknown condition: run the flow with simulated results
    MODEL, MODE_REASON = None, f"{type(e).__name__}: {e}"

app = FastAPI(title="Senyum API", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/meta")
def meta():
    metrics = RESULTS / "metrics.json"
    return {"mode": "model" if MODEL else "demo", "mode_reason": MODE_REASON, "order": ORDER, "conditions": CONDITIONS,
            "urgency": _content["urgency"], "emergency_signs": _content["emergency_signs"], "areas": sorted(tools.AREAS),
            "metrics": json.loads(metrics.read_text()) if metrics.exists() else None}


def findings(probs: list[float], thresholds: list[float]) -> dict:
    rows = [dict(id=c, probability=round(p, 3), threshold=t,
                 status="flagged" if p >= t else "borderline" if p >= t - BORDERLINE else "clear")
            for c, p, t in zip(ORDER, probs, thresholds)]
    flagged = [r["id"] for r in rows if r["status"] == "flagged"]
    urgency = max((CONDITIONS[c]["urgency"] for c in flagged), key=RANK.get, default="routine")
    return {"conditions": sorted(rows, key=lambda r: -r["probability"]), "flagged": flagged, "urgency": urgency}


@app.post("/api/screen")
async def screen(file: UploadFile | None = File(None), demo: str | None = Form(None)):
    if demo is not None or file is None:   # simulated result: for presenting the flow without a photo or without weights
        if demo is None:
            raise HTTPException(400, "No photo was sent.")
        chosen = set(demo.split(",")) & set(ORDER)
        return {"quality": {"ok": True, "issues": []}, "simulated": True,
                **findings([0.86 if c in chosen else 0.06 for c in ORDER], [0.5] * len(ORDER))}
    if not MODEL:
        raise HTTPException(503, "The trained model isn't loaded on this machine. Use a demo result instead.")
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
    out = {"quality": q, "simulated": False}
    if q["ok"]:
        pred = multilabel.predict(im, MODEL)["conditions"]
        out.update(findings([c["probability"] for c in pred], [c["threshold"] for c in pred]))
    del data, im
    return out


class PlanIn(BaseModel):
    conditions: list[str] = []
    profile: dict = {}
    area: str | None = None
    lat: float | None = None
    lng: float | None = None
    after_hours: bool = False


def estimate(conds: list[str], residency: str, tier: str, age: str) -> dict:
    """One estimate for several findings: a procedure shared by two findings (one cleaning visit) is counted once."""
    ests = [costs.estimate(CONDITIONS[c].get("cost_as", c), residency, tier, age) for c in conds]
    likely = list({l["id"]: l for e in ests for l in e["likely"]}.values())
    if_worse = [l for l in {l["id"]: l for e in ests for l in e["if_worse"]}.values() if l["id"] not in {x["id"] for x in likely}]
    return {"likely": likely, "if_worse": if_worse,
            "likely_total": {k: round(sum(l[k] for l in likely), 2) for k in ("fee_min", "fee_max", "pay_min", "pay_max")},
            "notes": list(dict.fromkeys(n for e in ests for n in e["notes"])),
            "sources": {k: v for e in ests for k, v in e["sources"].items()}, "binding": False}


@app.post("/api/plan")
def plan(body: PlanIn):
    """Rule-based agent: picks tools from the findings and the patient's profile, and records every call for the patient."""
    conds = [c for c in ORDER if c in body.conditions] or ["normal"]
    prof = body.profile
    residency, tier, age = prof.get("residency", "citizen"), prof.get("tier", "none"), prof.get("age", "under60")
    if body.lat is not None and body.lng is not None:
        lat, lng, place = body.lat, body.lng, "your location"
    else:
        area = tools.AREAS.get(body.area or "", tools.AREAS["Toa Payoh"])
        lat, lng, place = area["lat"], area["lng"], area["name"]
    chas = residency == "citizen" and tier not in ("none", "green")
    services = list(dict.fromkeys(p for c in conds for p in CONDITIONS[c]["procedures"]["likely"]))
    urgency = max((CONDITIONS[c]["urgency"] for c in conds), key=RANK.get)
    trace = [{"tool": "plan", "args": {"findings": conds, "urgency": urgency}, "ms": 0.0,
              "result": f"Needs a clinic offering {', '.join(services) or 'a general check-up'}; "
                        f"{'prefer CHAS clinics' if chas else 'no CHAS dental subsidy for this profile'}."}]

    def call(tool, args, fn, summarise):
        t0 = time.perf_counter()
        result = fn()
        trace.append({"tool": tool, "args": args, "result": summarise(result), "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return result

    cost = call("estimate_cost", {"findings": conds, "residency": residency, "tier": tier, "age": age},
                lambda: estimate(conds, residency, tier, age),
                lambda r: f"Likely S${r['likely_total']['pay_min']:,.2f} to S${r['likely_total']['pay_max']:,.2f} after subsidies"
                if r["likely"] else "No routine procedure fee for these findings")
    args = {"near": place, "services": services, "chas_eligible": chas, "after_hours": body.after_hours}
    clinics = call("find_clinics", args, lambda: tools.find_clinics(lat, lng, services, chas, body.after_hours),
                   lambda r: f"{len(r)} clinics, nearest {r[0]['distance_km']} km away" if r else "No matching clinics")
    if not clinics and body.after_hours:
        clinics = call("find_clinics", {**args, "after_hours": False, "retry": "relaxed opening hours"},
                       lambda: tools.find_clinics(lat, lng, services, chas, False),
                       lambda r: f"{len(r)} clinics after relaxing the opening-hours filter")
    wanted = {}   # product id -> first finding that asked for it
    for c in conds:
        for pid in CONDITIONS[c]["products"]:
            wanted.setdefault(pid, c)
    products = call("recommend_products", {"findings": conds},
                    lambda: [dict(id=pid, name=tools.PRODUCTS[pid]["name"], look_for=tools.PRODUCTS[pid]["look_for"],
                                  why=tools.PRODUCTS[pid]["why"].get(c, tools.PRODUCTS[pid]["why"]["default"])) for pid, c in wanted.items()],
                    lambda r: ", ".join(p["name"] for p in r))
    texts = [t for p in products for t in (p["why"], p["look_for"])] + [h for c in conds for h in CONDITIONS[c]["home"]]
    safety = call("safety_review", {"items_checked": len(texts)}, lambda: tools.safety_review(texts, urgency, False),
                  lambda r: "Passed: no medication or diagnostic wording" if r["passed"] else f"Flagged {len(r['flagged'])} item(s)")
    if not safety["passed"]:
        products = [p for p in products if not tools.BANNED.search(p["why"] + p["look_for"])]
    return {"planner": "rule-based", "near": place, "urgency": urgency, "trace": trace, "cost": cost, "clinics": clinics,
            "products": products, "safety": safety, "clinic_data": "sample"}


app.mount("/", StaticFiles(directory=HERE.parent / "frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SENYUM_PORT", "8730"))
    print("model: loaded" if MODEL else f"model: demo mode ({MODE_REASON})")
    print(f"Senyum: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
