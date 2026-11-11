# Oral condition classifier — web demo

Serves `models/classifier/weights/Daniel/binary/best.pt` — the Stage-1
**issue gate** — as a single-page web app: upload (or drag/drop / paste) an
intraoral photo and get a calibrated verdict. The demo reads the checkpoint's
own schema and adapts to whichever task it was trained for:

- **Issue gate** (`task: "binary"`, the default file): `p(issue)` with a flag
  when it is at or above the checkpoint's tuned `threshold` — the grey zone
  between 0.5 and the threshold is deliberately *not* flagged, so the verdict
  follows the threshold, not argmax.
- **Condition classifier** (6 classes, e.g. the Nov-5
  `weights/Daniel/best.pt`): per-class probabilities, top-2, and an abstention
  flag when confidence is under the checkpoint's `abstain_threshold`. Run it
  with `DEMO_WEIGHTS` pointed at that file.

The "About this model" strip (metrics table, bars, footer stats) is built from
the matching test report — `models/classifier/results/Daniel/test_metrics.json`
or `results/Daniel/binary/test_metrics.json`, chosen by the checkpoint's task —
so the numbers can never go stale against a new checkpoint.

## Mouth gate

Before the classifier runs, `mouth_gate.py` checks what the photo actually shows
(the classifier only ever saw intraoral close-ups):

| Route | When | What happens |
| --- | --- | --- |
| `intraoral` | close-up of teeth / gums / tongue | classified as-is |
| `face_crop` | a face photo with a big-enough, open mouth | mouth located, de-rotated, 4:3 crop sent to the model; the UI draws the box and shows the crop |
| `rejected` | not a mouth (`no_mouth`), face but mouth not found (`mouth_not_found`), mouth too small (`mouth_too_small`) or closed (`mouth_closed`) | classifier **not** run; the UI says why and how to retake |

- Scene type and mouth-open checks: CLIP zero-shot (`openai/clip-vit-base-patch16`,
  prompt groups in `SCENE_PROMPTS` / `OPEN_PROMPTS`). Mouth location: OpenCV YuNet
  face detector (mouth-corner landmarks), run at two scales.
- Thresholds live in `GateConfig`. Sanity check run on 2026-11-12: 595/600
  `oral_dataset` test images passed as intraoral (the 5 rejected were heavily
  blurred augmentations and one brochure scan); on 25 web photos, every face with
  a visible open mouth was cropped correctly, and books / distant people were rejected.
- First run downloads CLIP (~600 MB, Hugging Face cache) and YuNet (~230 KB, to
  `assets/`). If either fails to load, the demo prints a warning and classifies
  whole images as before. Adds ~40–100 ms per image on GPU.
- Caveat: crops from face photos are still further from the training data than
  true intraoral shots, so treat their scores with more caution.

## Run

```
pip install -r models/classifier/demo/Daniel/requirements.txt
python models/classifier/demo/Daniel/app.py
```

Then open **http://127.0.0.1:8712**. (On Windows you can double-click `run.bat`.)

Environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEMO_PORT` | `8712` | listen port |
| `DEMO_WEIGHTS` | `models/classifier/weights/Daniel/binary/best.pt` | try another checkpoint, e.g. the six-class `models\classifier\weights\Daniel\best.pt` |
| `DEMO_METRICS` | report folder chosen by task | point at a different `test_metrics.json` |
| `DEMO_MOUTH_GATE` | `1` | `0` disables the mouth gate (whole image goes to the model) |

## Details

- Preprocessing follows the checkpoint's own metadata (backbone, classes,
  head name, mean/std, image size, temperature), which is the same pipeline as
  the notebook's `predict_image()`: EXIF-orient → RGB → thumbnail ≤384 px →
  squash to 224×224 → ImageNet normalise → logits ÷ T → softmax.
- Checkpoints saved by `torch.save` of a plain dict load fine; if the strict
  `weights_only` unpickler rejects a file (torch ≥ 2.6 default), the demo
  retries with `weights_only=False`.
- GPU is used automatically if CUDA is available; CPU works fine (≈0.1 s/image
  on a laptop).
- Images are only held in memory during the request and are never written to
  disk or stored — the in-page history is cleared on reload.
- Endpoints: `GET /api/meta` (checkpoint + task + metrics; NaN literals in the
  report are sanitised to `null` so browsers can parse it), `POST /api/predict`
  (multipart field `file`). Interactive docs at `/api/docs`.
- The footer disclaimer is generated per task from the test report (threshold
  sensitivity/specificity/AUROC for the gate; macro-F1 and CI for the
  classifier). Screening/education aid only, not a diagnosis; gum disease is
  not a class.
