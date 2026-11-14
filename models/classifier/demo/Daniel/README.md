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

## Condition detector (stage 2)

When the classifier reports an anomaly — the binary gate flags the photo, or the
6-class checkpoint's top class is not `normal` — `detector.py` runs the YOLO model
chosen by `models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb`
(`models/yolo_caries/weights/Daniel/best.pt`, yolo11s) on the same image the
classifier saw (the intraoral photo, or the mouth crop from a face photo), at full
resolution rather than the classifier's 224 px input.

- Classes: Calculus, Caries, Hypodontia, Mouth_Ulcer, Tooth_Discoloration. Operating
  point, IoU and image size come from the `best.json` written next to the weights
  (`conf` 0.357 = the confidence that maximised mean F1 on val).
- The UI draws the boxes over the photo and lists one chip per condition (count and
  strongest confidence). Boxes between 0.15 and `conf` are returned as weaker
  candidates (`confident: false`), hidden behind a toggle.
- **A flagged photo is always outlined.** If nothing reaches `conf`, the detector's
  3 strongest non-overlapping guesses are returned however low their score (down to
  0.001; `guess: true` below 0.15) and drawn dashed, with a note that they are
  low-confidence hints. On the test set this happens for 6 of 387 flagged issue photos
  but 17 of 27 flagged Normal photos — a flagged photo with only best guesses is more
  often a classifier false alarm than a missed finding.
- Test numbers for the detector (mAP50 0.840, mAP50-95 0.450) are read from
  `models/yolo_caries/results/Daniel/test_metrics.json` into the "About this model"
  strip. Adds ~40–70 ms per flagged image on GPU; photos that aren't flagged skip it.
- `DEMO_DETECTOR=0` disables it; `DEMO_DETECTOR=<path to .pt>` tries other weights.
  If the weights or `ultralytics` are missing, the demo prints a warning and runs
  without it.

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
| `DEMO_DETECTOR` | `1` (`models/yolo_caries/weights/Daniel/best.pt`) | `0` disables the stage-2 detector, or a path to other YOLO weights |

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
- Endpoints: `GET /api/meta` (checkpoint + task + metrics + `detector`; NaN literals in the
  report are sanitised to `null` so browsers can parse it), `POST /api/predict`
  (multipart field `file`; adds `anomaly` and, when the detector ran, `detection`
  with `boxes` as `[x1, y1, x2, y2]` fractions of `detection.image`). Interactive docs at `/api/docs`.
- The footer disclaimer is generated per task from the test report (threshold
  sensitivity/specificity/AUROC for the gate; macro-F1 and CI for the
  classifier). Screening/education aid only, not a diagnosis; gum disease is
  not a class.
