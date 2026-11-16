"""Out-of-pocket dental cost estimate after CHAS and Flexi-MediSave.

Adapted from Daniel's prototype/Daniel/backend/costs.py. Changes:
- fees.json covers more procedures, every CHAS tier, CHAS claim limits and 2025 hospital averages,
  all checked against the official MOH / CHAS pages (see `sources` in fees.json).
- GST is applied the way CHAS says: GST-registered clinics add 9% GST *before* the CHAS subsidy.
  The range runs from the cheapest case (lowest fee, no GST) to the dearest (highest fee + GST).
- "later" holds alternative treatments (e.g. denture OR implant), each costed on its own.

Every figure is tied to a source. A non-binding estimate, never a quote.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
FEES = json.loads((DATA / "fees.json").read_text(encoding="utf-8"))

RESIDENCY = {"citizen", "pr", "foreigner"}
TIERS = {"pioneer", "merdeka", "blue", "orange", "green", "none", "unknown"}
AGE_GROUPS = {"under60", "60plus"}

# Accept both the model's names (shared/conditions.yaml) and the front-end's short keys.
ALIASES = {"discolouration": "tooth_discoloration", "ulcer": "mouth_ulcer", "pass": "foreigner"}


def _norm(value: str | None, allowed: set[str], default: str) -> str:
    v = ALIASES.get((value or "").strip().lower(), (value or "").strip().lower())
    return v if v in allowed else default


def _chas_subsidy(proc: dict, residency: str, tier: str) -> float:
    """Maximum CHAS subsidy for one procedure (sum over its CHAS items)."""
    chas = FEES["chas"]
    if residency not in chas["eligible_residency"] or tier not in chas["dental_tiers"]:
        return 0.0
    return sum(chas["max_per_claim"].get(item, {}).get(tier, 0.0) for item in proc.get("chas_items", []))


def _line(pid: str, profile: dict, medisave_left: dict) -> dict:
    """Cost one procedure. `medisave_left` = remaining Flexi-MediSave cap for the low and high case."""
    proc = FEES["procedures"][pid]
    gst = FEES["gst_rate"]
    fee_min, fee_max = float(proc["min"]), float(proc["max"])
    sub = _chas_subsidy(proc, profile["residency"], profile["tier"])

    # Cheapest case: lowest fee at a clinic without GST. Dearest: highest fee + GST, then subsidy.
    after_low = max(0.0, fee_min - sub)
    after_high = max(0.0, fee_max * (1 + gst) - sub)

    medi_low = medi_high = 0.0
    flexi = FEES["flexi_medisave"]
    if (proc.get("flexi_medisave") and profile["age"] == flexi["min_age_group"]
            and profile["residency"] in flexi["eligible_residency"]):
        medi_low = min(medisave_left["low"], after_low)
        medi_high = min(medisave_left["high"], after_high)
        medisave_left["low"] -= medi_low
        medisave_left["high"] -= medi_high

    sources = [proc["source"]]
    if sub:
        sources.append(FEES["chas"]["source"])
    if medi_high:
        sources.append(flexi["source"])

    limits = FEES["chas"]["claim_limits"]
    claim_limit = "; ".join(dict.fromkeys(limits[i] for i in proc.get("chas_items", []) if i in limits)) if sub else None

    return {
        "id": pid,
        "name": proc["name"],
        "plain": proc["plain"],
        "fee_min": fee_min,
        "fee_max": fee_max,
        "subsidy": round(sub, 2),
        "medisave": round(medi_high, 2),
        "pay_min": round(after_low - medi_low, 2),
        "pay_max": round(after_high - medi_high, 2),
        "claim_limit": claim_limit,
        "range_note": proc.get("range_note"),
        "sources": sources,
    }


def _group(pids: list[str], profile: dict) -> dict:
    cap = float(FEES["flexi_medisave"]["annual_cap"])
    left = {"low": cap, "high": cap}  # the S$400 cap is shared by everything in one plan
    lines = [_line(p, profile, left) for p in pids]
    total = {k: round(sum(l[k] for l in lines), 2)
             for k in ("fee_min", "fee_max", "subsidy", "medisave", "pay_min", "pay_max")}
    return {"lines": lines, "total": total}


def estimate(condition: str, residency: str = "citizen", tier: str = "unknown", age: str = "under60") -> dict:
    condition = ALIASES.get(condition, condition)
    if condition not in FEES["plans"]:
        raise ValueError(f"unknown condition {condition!r}")

    residency = _norm(residency, RESIDENCY, "citizen")
    tier = _norm(tier, TIERS, "unknown")
    age = _norm(age, AGE_GROUPS, "under60")
    profile = {"residency": residency, "tier": tier, "age": age}

    plan = FEES["plans"][condition]
    likely = _group(plan["likely"], profile)
    later = [{"label": s["label"], **_group(s["procedures"], profile)} for s in plan["later"]]

    # Notes most relevant to this person go first.
    notes = []
    if residency != "citizen":
        notes.append("CHAS dental subsidies are for Singapore Citizens only.")
    elif tier == "green":
        notes.append("CHAS Green cards don't include dental subsidies.")
    elif tier in ("unknown", "none"):
        notes.append("No CHAS subsidy is included. Check your card at chas.sg: Blue, Orange, Merdeka and Pioneer cards lower these prices.")
    if any(l["medisave"] for g in [likely, *later] for l in g["lines"]):
        notes.append(FEES["flexi_medisave"]["note"])
    if plan.get("note"):
        notes.append(plan["note"])
    notes.extend(FEES["general_notes"])

    used = {s for g in [likely, *later] for l in g["lines"] for s in l["sources"]}
    if residency != "citizen" or tier == "green":
        used.add("chas_faq")
    used.add("chas_dental")  # claim limits + the GST-before-subsidy rule behind every range

    return {
        "condition": condition,
        "profile": profile,
        "likely": likely,
        "later": later,
        "notes": notes,
        "sources": {k: v for k, v in FEES["sources"].items() if k in used},
        "gst_rate": FEES["gst_rate"],
        "binding": False,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(estimate(*sys.argv[1:]), indent=2))
