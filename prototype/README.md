# SmileCheck SG: front-end prototype

A patient-facing web page that covers the Detection, Understanding and Action gaps from the brief, with Cost as a placeholder. There is no backend: the AI result is simulated in `predict()` in `js/app.js`, which is where the trained CNN plugs in.

## Run it

```bash
python -m http.server 5173 --bind 127.0.0.1
```

Run it from this folder, then open http://127.0.0.1:5173/. Opening `index.html` directly also works, but "Use my location" needs localhost or https.

Jump straight to a result with `?demo=calculus`, `caries`, `discolouration`, `ulcer`, `hypodontia` or `normal`.

## Files

| File | What it holds |
| --- | --- |
| `index.html` | Page structure: hero, screening, explainer, cost placeholder, clinics and care |
| `css/styles.css` | Neumorphic design tokens and components |
| `js/data.js` | Plain-language content per condition, products, clinics, areas |
| `js/art.js` | Hand-drawn SVG teeth, conditions, photo angles and products |
| `js/app.js` | Upload and consent, photo-quality check, simulated model, results, Leaflet map |

## Prototype data to replace

- `predict()` returns simulated likelihoods. Replace it with the real model call.
- The three "CHAS dental clinic" entries are samples. The agent will fetch live CHAS clinics.
- Clinic coordinates are approximate and only cover the north-east, and the HealthHub link goes to the homepage.
- The cost estimator is a placeholder.
