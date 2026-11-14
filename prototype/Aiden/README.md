# Senyum: mouth photo check (Aiden's prototype)

A patient-facing web app for the four gaps in the IT2226 brief. One photo can come back with several findings (for
example tartar *and* discolouration), and the cost and clinic steps cover all of them together. The findings are meant
to come from Aiden's stage-1 gate + YOLO detector (see "Not wired to the gate yet" below).

| Step | Gap | What happens |
|---|---|---|
| 1. Photo | Detection | Emergency check (995 / A&E), PDPA consent, upload or camera, photo-quality gate |
| 2. Findings | Understanding | A score per condition against its own threshold (flagged / borderline / not flagged), plain-language explanation and urgency for each flagged one |
| 3. Cost & clinics | Cost, Action | Rule-based agent: one estimate across all findings after CHAS and Flexi-MediSave (shared procedures counted once, every figure sourced and non-binding), nearby clinics, home-care products, safety review. Its tool trace is shown to the patient |

## Run

```bash
.venv\Scripts\activate
pip install fastapi uvicorn python-multipart      # torch, torchvision and pillow are already in the training venv
python prototype/Aiden/backend/app.py
```

Open **http://127.0.0.1:8730** (`SENYUM_PORT` changes the port).

**Not wired to the gate yet.** Since 2026-11-12 the trained classifier is a binary gate (healthy vs has an issue) and
the per-condition findings this page shows must come from the YOLO detector, which isn't built yet. Until then the
backend refuses the gate checkpoint (no per-condition classes) and runs in demo mode.

**Model vs demo mode.** The backend loads `models/classifier/results/Aiden/<model>/model.pt` (written by
`models/classifier/scripts/Aiden/train.py`). `<model>` is `dinov2_vitb14` by default; set `SENYUM_MODEL=densenet121` to
serve DenseNet instead. Without it the header shows "Demo mode"
and only simulated results work: "No photo? Walk through a simulated result", or a deep link such as
`?demo=calculus,tooth_discoloration` (add `&plan` to run step 3 too). Simulated results are always labelled as such.

## Layout

```
backend/app.py            FastAPI: /api/meta, /api/screen, /api/plan; serves ../frontend
frontend/                 index.html, styles.css, app.js (no build step, no external assets)
```

## Shared with Daniel's prototype

`backend/app.py` **imports** (does not copy or change) `costs.py`, `quality.py`, the clinic / product / guardrail tools in
`agent.py`, and the data files from `prototype/Daniel/backend/`. So the same caveats apply: clinics are sample listings,
fees are only the figures quoted in the brief, and an unsourced CHAS amount shows "Ask clinic". If Daniel renames those
functions, this prototype breaks; the shared code should move to `cost/` and `genai/agent/` when the prototypes merge.

## Not done yet

- The planner is rule-based. An LLM planner would replace `plan()` in `app.py`; tools and trace format can stay.
- English only. No map. No real CHAS clinic lookup.
- No automated tests: endpoints were checked by hand (see `docs/test_log.md` conventions when adding scenarios).
