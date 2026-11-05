"""Out-of-pocket estimate after CHAS and Flexi-MediSave.

Only uses figures in data/fees.json, each tied to a source. When a subsidy amount is
not in the sourced table, the line is flagged `subsidy_unknown` instead of guessed.
All output is a non-binding estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
FEES = json.loads((DATA / "fees.json").read_text(encoding="utf-8"))
CONDITIONS = json.loads((DATA / "conditions.json").read_text(encoding="utf-8"))["conditions"]

RESIDENCY = {"citizen", "pr", "pass"}
TIERS = {"pioneer", "merdeka", "blue", "orange", "green", "none"}
AGE_GROUPS = {"under60", "60plus"}


def _chas_amount(proc: dict, residency: str, tier: str) -> tuple[float, bool]:
    """Return (max subsidy per visit, known?)."""
    chas = FEES["chas"]
    if residency not in chas["eligible_residency"] or tier in ("none", "green"):
        return 0.0, True
    items = proc.get("chas_items") or []
    if not items:
        return 0.0, True
    total, known = 0.0, True
    for item in items:
        table = chas["max_per_claim"].get(item, {})
        if tier in table:
            total += table[tier]
        else:
            known = False
    return total, known


def _line(pid: str, residency: str, tier: str, age: str, medisave_left: list[float]) -> dict:
    proc = FEES["procedures"][pid]
    lo, hi = float(proc["min"]), float(proc["max"])
    sub, known = _chas_amount(proc, residency, tier)

    medi = 0.0
    flexi = FEES["flexi_medisave"]
    if proc.get("medisave_flexi") and age == flexi["min_age_group"] and residency in flexi["eligible_residency"]:
        medi = min(medisave_left[0], max(0.0, hi - sub))
        medisave_left[0] -= medi

    def net(x: float) -> float:
        after_sub = max(0.0, x - sub)
        return round(max(0.0, after_sub - min(medi, after_sub)), 2)

    sources = [proc["source"]]
    if sub:
        sources.append(FEES["chas"]["source"])
    if medi:
        sources.append(flexi["source"])
    return {
        "id": pid,
        "name": proc["name"],
        "plain": proc["plain"],
        "fee_min": lo, "fee_max": hi,
        "subsidy": round(sub, 2),
        "subsidy_unknown": not known,
        "medisave": round(medi, 2),
        "pay_min": net(lo), "pay_max": net(hi),
        "sources": sources,
    }


def estimate(condition: str, residency: str = "citizen", tier: str = "none", age: str = "under60") -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    residency = residency if residency in RESIDENCY else "citizen"
    tier = tier if tier in TIERS else "none"
    age = age if age in AGE_GROUPS else "under60"
    if residency != "citizen":
        tier = "none"

    plan = CONDITIONS[condition]["procedures"]
    medisave_left = [float(FEES["flexi_medisave"]["annual_cap"])]
    likely = [_line(p, residency, tier, age, medisave_left) for p in plan["likely"]]
    if_worse = [_line(p, residency, tier, age, medisave_left) for p in plan["if_worse"]]

    def total(lines: list[dict], key: str) -> float:
        return round(sum(l[key] for l in lines), 2)

    notes = list(FEES["general_notes"])
    if residency == "citizen" and tier in ("pioneer", "merdeka", "blue", "orange"):
        notes.insert(0, FEES["chas"]["claims_note"])
    if residency == "citizen" and tier == "green":
        notes.insert(0, "CHAS Green cards don't include dental subsidies.")
    if residency != "citizen":
        notes.insert(0, "CHAS dental subsidies are for Singapore Citizens only.")
    if any(l["medisave"] for l in likely + if_worse):
        notes.insert(0, FEES["flexi_medisave"]["note"])
    if any(l["subsidy_unknown"] for l in likely + if_worse):
        notes.insert(0, FEES["chas"]["unknown_note"])

    used = {s for l in likely + if_worse for s in l["sources"]}
    return {
        "condition": condition,
        "profile": {"residency": residency, "tier": tier, "age": age},
        "likely": likely,
        "if_worse": if_worse,
        "likely_total": {"fee_min": total(likely, "fee_min"), "fee_max": total(likely, "fee_max"),
                         "pay_min": total(likely, "pay_min"), "pay_max": total(likely, "pay_max")},
        "notes": notes,
        "sources": {k: v for k, v in FEES["sources"].items() if k in used},
        "gst_rate": FEES["gst_rate"],
        "binding": False,
    }
