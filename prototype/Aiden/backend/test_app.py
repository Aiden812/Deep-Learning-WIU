"""Self-check: python prototype/Aiden/backend/test_app.py"""
import app

# two findings that share a cleaning are charged one cleaning, not two
one, both = app.estimate(["calculus"], "citizen", "blue", "under60"), app.estimate(["calculus", "tooth_discoloration"], "citizen", "blue", "under60")
assert [l["id"] for l in both["likely"]] == ["scale_polish"] and both["likely_total"] == one["likely_total"]
assert one["likely_total"]["pay_max"] == 98 - 30 - 20.5   # brief: CHAS Blue scaling 30.00 + polishing 20.50

# a procedure already in the likely plan is not repeated under "if it has progressed"
mixed = app.estimate(["caries", "calculus"], "pr", "none", "60plus")
assert not {l["id"] for l in mixed["likely"]} & {l["id"] for l in mixed["if_worse"]}

f = app.findings([0.9, 0.4, 0.1, 0.6, 0.1], [0.5] * 5)
assert f["flagged"] == ["calculus", "mouth_ulcer"] and f["urgency"] == "watch"
assert {c["id"]: c["status"] for c in f["conditions"]}["caries"] == "borderline"
assert app.findings([0.0] * 5, [0.5] * 5)["urgency"] == "routine"

plan = app.plan(app.PlanIn(conditions=["caries"], area="Bedok"))
assert plan["safety"]["passed"] and plan["clinics"] and plan["products"]
print("ok")
