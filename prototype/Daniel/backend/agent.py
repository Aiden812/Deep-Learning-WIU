"""Action agent: turns a screening result into next steps for the patient.

The planner is rule-based: it picks tools from the screening result and patient profile,
runs them, and records every call in a trace the UI shows to the patient
("What Gigi did"). Tools:

    estimate_cost       -> costs.estimate
    find_clinics        -> ranks clinics by distance, services and CHAS acceptance
    recommend_products  -> home-care products matched to the condition
    safety_review       -> blocks medication or diagnostic wording, adds escalation advice

A language-model planner can replace `plan()` later. The tools and trace format stay the same.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import costs

DATA = Path(__file__).resolve().parent / "data"
CONDITIONS = json.loads((DATA / "conditions.json").read_text(encoding="utf-8"))
CLINICS = json.loads((DATA / "clinics.json").read_text(encoding="utf-8"))["clinics"]
PRODUCTS = json.loads((DATA / "products.json").read_text(encoding="utf-8"))["products"]
AREAS = {a["name"]: a for a in json.loads((DATA / "areas.json").read_text(encoding="utf-8"))["areas"]}

# Guardrail: the product must not recommend medication or state a diagnosis.
BANNED = re.compile(r"\b(antibiotic|ibuprofen|paracetamol|painkiller|steroid|chlorhexidine|dosage|mg\b|"
                    r"prescri\w*|you have|diagnos\w*)", re.I)


def _km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---- tools -----------------------------------------------------------------

def find_clinics(lat: float, lng: float, services: list[str], chas_eligible: bool,
                 after_hours: bool, limit: int = 4) -> list[dict]:
    out = []
    for c in CLINICS:
        missing = [s for s in services if s not in c["services"]]
        if missing:
            continue
        if after_hours and not (c["evenings"] or c["weekends"]):
            continue
        d = _km(lat, lng, c["lat"], c["lng"])
        # Score: distance first. CHAS clinics get a 1.5 km head start for subsidy-eligible
        # patients, since subsidies only apply at CHAS clinics.
        score = d - (1.5 if chas_eligible and c["chas"] else 0)
        out.append({**c, "distance_km": round(d, 1), "_score": score})
    out.sort(key=lambda c: c["_score"])
    for c in out:
        c.pop("_score", None)
        why = []
        if chas_eligible and c["chas"]:
            why.append("Accepts CHAS")
        if c["setting"] == "polyclinic":
            why.append("Lower fees, longer waits")
        if c["evenings"]:
            why.append("Evening hours")
        if c["weekends"]:
            why.append("Open weekends")
        c["why"] = why
    return out[:limit]


def recommend_products(condition: str) -> list[dict]:
    items = []
    for pid in CONDITIONS["conditions"][condition]["products"]:
        p = PRODUCTS[pid]
        items.append({"id": pid, "name": p["name"], "look_for": p["look_for"],
                      "why": p["why"].get(condition, p["why"]["default"])})
    return items


def safety_review(texts: list[str], urgency: str, abstain: bool) -> dict:
    flagged = [t for t in texts if BANNED.search(t)]
    advice = []
    if abstain:
        advice.append("The screening wasn't confident, so a dentist should take a look in person.")
    if urgency == "soon":
        advice.append("Book within 2 to 4 weeks. Don't wait for pain.")
    advice.append("If you develop facial swelling, bleeding that won't stop, or a dental injury, call 995 or go to A&E.")
    return {"passed": not flagged, "flagged": flagged, "advice": advice}


# ---- planner ---------------------------------------------------------------

def run(req: dict) -> dict:
    condition = req.get("condition", "normal")
    if condition not in CONDITIONS["conditions"]:
        raise ValueError(f"unknown condition {condition!r}")
    abstain = bool(req.get("abstain"))
    profile = req.get("profile") or {}
    residency, tier, age = profile.get("residency", "citizen"), profile.get("tier", "none"), profile.get("age", "under60")
    after_hours = bool(req.get("after_hours"))

    if req.get("lat") is not None and req.get("lng") is not None:
        lat, lng, place = float(req["lat"]), float(req["lng"]), "your location"
    else:
        area = AREAS.get(req.get("area") or "", AREAS["Toa Payoh"])
        lat, lng, place = area["lat"], area["lng"], area["name"]

    info = CONDITIONS["conditions"][condition]
    urgency = "unsure" if abstain else info["urgency"]
    trace: list[dict] = []

    def call(tool: str, args: dict, fn, summarise):
        t0 = time.perf_counter()
        result = fn()
        trace.append({"tool": tool, "args": args, "result": summarise(result),
                      "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return result

    # 1. Decide what the patient needs.
    services = list(info["procedures"]["likely"])
    trace.append({"tool": "plan", "args": {"condition": condition, "abstain": abstain, "urgency": urgency},
                  "result": (f"Needs a clinic offering {', '.join(services) or 'a general check-up'}; "
                             f"{'prefer CHAS clinics' if residency == 'citizen' and tier not in ('none', 'green') else 'no CHAS dental subsidy for this profile'}."),
                  "ms": 0.0})

    # 2. Cost.
    cost = call("estimate_cost", {"condition": condition, "residency": residency, "tier": tier, "age": age},
                lambda: costs.estimate(condition, residency, tier, age),
                lambda r: (f"Likely S${r['likely_total']['pay_min']:,.2f} to S${r['likely_total']['pay_max']:,.2f} after subsidies"
                           if r["likely"] else "No routine procedure fee for this result"))

    # 3. Clinics.
    chas_eligible = residency == "citizen" and tier not in ("none", "green")
    clinics = call("find_clinics", {"near": place, "services": services, "chas_eligible": chas_eligible,
                                    "after_hours": after_hours},
                   lambda: find_clinics(lat, lng, services, chas_eligible, after_hours),
                   lambda r: (f"{len(r)} clinics, nearest {r[0]['distance_km']} km away" if r else "No matching clinics"))
    if not clinics and after_hours:
        clinics = call("find_clinics", {"near": place, "services": services, "chas_eligible": chas_eligible,
                                        "after_hours": False, "retry": "relaxed opening hours"},
                       lambda: find_clinics(lat, lng, services, chas_eligible, False),
                       lambda r: f"{len(r)} clinics after relaxing the opening-hours filter")

    # 4. Products.
    products = call("recommend_products", {"condition": condition},
                    lambda: recommend_products(condition),
                    lambda r: ", ".join(p["name"] for p in r))

    # 5. Guardrail over everything shown to the patient.
    texts = [p["why"] for p in products] + [p["look_for"] for p in products] + info["home"]
    safety = call("safety_review", {"items_checked": len(texts)},
                  lambda: safety_review(texts, urgency, abstain),
                  lambda r: "Passed: no medication or diagnostic wording" if r["passed"] else f"Flagged {len(r['flagged'])} item(s)")
    if not safety["passed"]:
        products = [p for p in products if not BANNED.search(p["why"] + p["look_for"])]

    return {
        "planner": "rule-based",
        "near": place,
        "urgency": urgency,
        "trace": trace,
        "cost": cost,
        "clinics": clinics,
        "products": products,
        "safety": safety,
        "clinic_data": "sample",
    }
