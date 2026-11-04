# Oral condition classifier — web demo

Serves `models/classifier/weights/Daniel/best.pt` as a single-page web app:
upload (or drag/drop / paste) an intraoral photo and get the calibrated
probabilities for all six classes in `shared/conditions.yaml`, the top-2
prediction, an abstention flag, and per-class F1 from the test report.

## Run

```
pip install -r models/classifier/demo/Daniel/requirements.txt
python models/classifier/demo/Daniel/app.py
```

Then open **http://127.0.0.1:8712**. (On Windows you can double-click `run.bat`.)
Set `DEMO_PORT=...` before launching to use another port.

## Details

- Inference follows the checkpoint's own metadata (backbone, classes, mean/std,
  temperature, abstain threshold), which is the same pipeline as the notebook's
  `predict_image()`: EXIF-orient → RGB → thumbnail ≤384 px → squash to 224×224
  → ImageNet normalise → logits ÷ T → softmax.
- GPU is used automatically if CUDA is available; CPU works fine (≈0.1 s/image
  on a laptop).
- Images are only held in memory during the request and are never written to
  disk or stored — the in-page history is cleared on reload.
- Endpoints: `GET /api/meta` (checkpoint + metrics), `POST /api/predict`
  (multipart field `file`). Interactive docs at `/api/docs`.
- Disclaimer text shown in the UI comes from the checkpoint's `notes` field:
  screening/education aid only, not a diagnosis; gum disease is not a class.
