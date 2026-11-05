# Gigi: oral health check (Daniel's prototype)

A patient-facing web app that closes the four gaps in the IT2226 brief in one flow:

| Step | Gap | What happens |
|---|---|---|
| 1. Photo | Detection | Emergency check (995 / A&E), PDPA consent, upload or camera |
| 2. What it might be | Understanding | Photo-quality gate, classifier result with confidence, full score spread, abstention when unsure, plain-language explanation and urgency |
| 3. Cost | Cost | Estimate after CHAS tier, residency, age and Flexi-MediSave; every figure footnoted to its source and marked non-binding |
| 4. Clinics & care | Action | Agent plans, estimates cost, finds clinics, matches home-care products and runs a safety review. Its tool trace is shown to the patient |

## Run

```bash
cd prototype/Daniel/backend
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8720**. On Windows you can instead double-click `prototype/Daniel/run.bat`, which creates the venv on first run. Set `GIGI_PORT` to change the port. API docs are at `/api/docs`.

### Demo mode vs the real model

If `torch` and `torchvision` are installed **and** `models/classifier/weights/Daniel/best.pt` exists (or `GIGI_WEIGHTS` points at a checkpoint), the backend runs the trained classifier. It uses the same preprocessing, temperature scaling and abstain threshold as `models/classifier/demo/Daniel/app.py`.

Otherwise it runs in **demo mode**. The header shows a "Demo mode" pill and results carry a "Simulated result" tag. In demo mode you can pick a scenario, or use "No photo? Run a demo" to present the flow without a real photo. `?demo=caries` preselects a scenario.

## Layout

```
backend/
  app.py          FastAPI app; serves ../frontend at /
  classifier.py   real model loader + labelled demo fallback
  quality.py      too small / dark / bright / blurry / low-detail checks
  costs.py        fee + CHAS + Flexi-MediSave estimator (sourced figures only)
  agent.py        rule-based planner + tools + guardrail, returns a trace
  data/           conditions.json, fees.json, products.json, clinics.json, areas.json
frontend/
  index.html, css/styles.css, js/app.js, assets/mark.svg
PRODUCT.md        design context (users, principles, anti-references)
```

## Prototype data to replace

- `data/clinics.json` holds **sample clinics, not real ones**. The UI labels them "Sample listing". A production `find_clinics` tool should query the CHAS clinic locator.
- `fees.json` uses only figures quoted in the brief. Where a CHAS amount isn't sourced (e.g. fillings, root canals, non-Pioneer crowns), the estimate says "Ask clinic" rather than guessing. Source URLs for MOH and CPF point at the agency home pages and should be replaced with the exact pages.
- The agent planner is rule-based. `agent.plan`/`run` is the seam for an LLM planner; the tools and trace format can stay as they are.

## Privacy

Photos are read into memory, screened and discarded. Nothing is written to disk or logged, and responses are sent with `Cache-Control: no-store`. Profile answers and location stay in the browser tab.
