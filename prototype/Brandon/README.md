# SmileCheck SG: front-end prototype

A patient-facing web page that covers the Detection, Understanding and Action gaps from the brief, with Cost as a placeholder. Uploaded photos go to the real condition classifier (`models/classifier`) through the backend in `app/backend/main.py`. The "Try a sample" chips still show simulated results.

## Run it

Needs Python 3.10–3.13 (TensorFlow doesn't support 3.14 yet), and `oral_classifier.keras` plus `class_names.json` in `models/classifier/weights/`. From the repo root:

```bash
py -3.13 -m venv .venv
.venv\Scripts\pip install -r app/requirements.txt
.venv\Scripts\python -m uvicorn app.backend.main:app --port 8000
```

Then open http://127.0.0.1:8000/. The backend serves this page and the `/api/predict` endpoint.

Jump straight to a sample result with `?demo=calculus`, `caries`, `discolouration`, `ulcer`, `hypodontia` or `normal`.

## Files

| File | What it holds |
| --- | --- |
| `index.html` | Page structure: hero, screening, explainer, cost placeholder, clinics and care |
| `css/styles.css` | Neumorphic design tokens and components |
| `js/data.js` | Plain-language content per condition, products, clinics, areas |
| `js/art.js` | Hand-drawn SVG teeth, conditions, photo angles and products |
| `js/app.js` | Upload and consent, photo-quality check, model call, results, Leaflet map |

## Prototype data to replace

- The three "CHAS dental clinic" entries are samples. The agent will fetch live CHAS clinics.
- Clinic coordinates are approximate and only cover the north-east, and the HealthHub link goes to the homepage.
- The cost estimator is a placeholder.
