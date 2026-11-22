# SmileCheck SG: front-end prototype

A patient-facing web page that covers the Detection, Understanding and Action gaps from the brief, with Cost as a one-line placeholder. Photos and videos (uploaded, dropped, or pasted with Ctrl+V) go through Daniel's models, using the same pipeline as the web demo in [`models/classifier/demo/Daniel`](../../models/classifier/demo/Daniel/README.md). The "Try a sample" chips still show simulated results.

## Run it

```bash
pip install -r models/classifier/demo/Daniel/requirements.txt
python prototype/Daniel/server.py
```

Run these from the repo root, or double-click `run.bat`. Then open http://127.0.0.1:8713/. `server.py` serves this page and the `/api/predict` and `/api/meta` endpoints.

Jump straight to a sample result with `?demo=calculus`, `caries`, `discolouration`, `ulcer`, `hypodontia`, `normal` or `check`.

## How a screening works

`server.py` imports the demo's `app.py` instead of copying it, so both pages always run the same models, thresholds and preprocessing. There is no limit on how many photos or videos you add. Every photo, and every frame of every video, goes through:

1. **Mouth gate** (`mouth_gate.py`): a close-up is used as it is. For a face photo, the gate crops to the mouth. A photo with no usable mouth is skipped and the page says why (no mouth, mouth closed, too small).
2. **Issue gate** (`weights/Daniel/binary/best.pt`): p(issue), flagged at the checkpoint's tuned threshold (0.73).
3. **YOLO detector** (`yolo_caries/weights/Daniel/best.pt`): runs only on flagged photos and outlines calculus, caries, hypodontia, mouth ulcers and discolouration.

### Turning the classifier off

To screen with YOLO alone, skip step 2 in either of these ways:

- **For the whole server:** set `PROTO_CLASSIFIER=0` before starting it (PowerShell: `$env:PROTO_CLASSIFIER = "0"; python prototype/Daniel/server.py`).
- **For one page load:** open http://127.0.0.1:8713/?classifier=off. Use `?classifier=on` to force the classifier back on when the server runs with `PROTO_CLASSIFIER=0`. The page sends this as the `classifier` form field of `/api/predict`.

With the classifier off, YOLO runs on every photo and frame that passes the mouth gate:

- A photo or frame is flagged when it has a box at or above the detector's operating point (`conf` in `best.json`, 0.361 for the current model). Video and multi-photo results are combined with the same rules as below.
- The score is the strongest box's confidence. It is not a calibrated probability, so the ring text changes to say how strongly the clearest finding was outlined.
- No best guesses are forced onto a photo. A photo YOLO finds clean shows no boxes, and weaker boxes on a healthy result aren't drawn.
- The classifier is still loaded at startup, because the demo module loads it, but it isn't run. The server won't start with `PROTO_CLASSIFIER=0` if the detector can't load.

Videos are decoded with OpenCV from a temporary file that is deleted when the request ends. Videos longer than `VIDEO_MAX_FRAMES` (default 1800, one minute at 30 fps) are sampled evenly down to that many frames. `/api/predict` streams NDJSON (a start line, one progress line per photo or frame, then the result), so the page shows a frame counter while a video runs.

The results are then combined per source (each photo or video), then across sources:

- **Video verdict:** a video counts as flagged when at least 10% of its readable frames are flagged (`VIDEO_FLAG_FRACTION`), and at least 2 frames, so one blurry frame can't flag a whole video. Its score is the k-th highest frame p(issue), which is at or above the threshold exactly when that many frames are flagged.
- **Verdict:** if any photo or video is flagged, the screening says *Needs a check*, because an issue may show from only one angle. The ring shows the highest score. If nothing is flagged, it says *Looks healthy*, and the ring shows 1 − p(issue).
- **Condition:** the strongest detector finding sets the condition shown and the Understand and Clinics sections. If it isn't confident, the name says "Might be" and the boxes are dashed. If the detector finds nothing, the page shows the generic *Worth a check* result.
- **Bars:** each condition's strongest box confidence, pooled over the flagged photos and videos.
- **Hero photo:** the photo or video frame with the strongest finding, drawn with its boxes. Only that image is sent back, so long videos stay light.
- **Notes:** skipped photos say why; each video gets a line like "60 of 60 frames checked, 45 worth a check".

The demo's environment overrides apply here too: `DEMO_WEIGHTS` (the six-class `weights/Daniel/best.pt` also works, with probabilities averaged within each photo or video, then across them), `DEMO_MOUTH_GATE=0` and `DEMO_DETECTOR=0`. `VIDEO_MAX_FRAMES`, `VIDEO_FLAG_FRACTION` and `VIDEO_MAX_MB` (default 300) tune video handling. `PROTO_PORT` changes the port. If you change it, the page has to be opened from that same server.

## Files

| File | What it holds |
| --- | --- |
| `server.py` | Backend: serves the page, runs each photo and video frame through the demo's pipeline, streams progress and combines the results |
| `run.bat` | Starts `server.py` on Windows |
| `index.html` | Page structure: hero, screening, then (after a result) explainer and clinics and care |
| `css/styles.css` | Neumorphic design tokens and components |
| `js/data.js` | Plain-language content per condition, products, clinics, areas |
| `js/art.js` | Hand-drawn SVG teeth, conditions and products |
| `js/app.js` | Upload, paste and consent, photo-quality check, streamed model call, detector boxes, results, Leaflet map |

## Prototype data to replace

- The three "CHAS dental clinic" entries are samples. The agent will fetch live CHAS clinics.
- Clinic coordinates are approximate and only cover the north-east, and the HealthHub link goes to the homepage.
- The cost estimator is a placeholder.
