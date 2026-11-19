# External benchmark

Scores every team model on photos **none of them trained on**. Plan and dataset audit: `docs/benchmark_plan.md`.
Results: `results/benchmark_report.md`.

## What is in it
| Dataset | Bench (scored) | Train part (may be trained on later) | Tests |
|---|---|---|---|
| AlphaDent (dentist caries outlines, clinic DSLR) | valid + 100 patients | other ~170 patients | caries boxes, caries yes/no |
| MetaDent DS1 hospital + DS3 web (dentist labels) | all DS1 + half of DS3 | other half of DS3 | problem vs healthy, each condition |
| Autooral (ulcer close-ups + masks) | Val + Test + 1/3 of train patients | rest | ulcer recall, ulcer boxes |
| SMART-OM normal tongue / cheek / lip | half the patients | other half | ulcer false alarms |

Dropped: MetaDent DS2 (same Mendeley photos as our Kaggle training data), Caries-Spectra (augmented single-tooth
crops), SMART-OM arch photos (teeth not labelled), and any photo matching a team training photo
(pHash <= 6 or >= 25 SIFT keypoints). `manifest.csv` lists every photo, its role, and why it was excluded.

**Rule: never train on a `bench` row.** Train only on `role == train` rows, so later models stay comparable.

## Run it (from the workspace folder that holds `.venv-train/`)
```
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/dedupe_phash.py     # copy check 1 (cached)
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/dedupe_sift.py      # copy check 2 (cached)
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/build_manifest.py   # photos, labels, split
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/sanity_check.py     # adapters reproduce notebook numbers
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py all          # predictions (resumes)
Deep-Learning-WIU/.venv/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py team_keras_6class
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py all --speed  # CPU timing
.venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/score.py            # report
```
The raw datasets must be in `data/raw/alpha-dent/` and `data/raw/external/` (see `docs/benchmark_plan.md`).

## Adding your model (Aiden, Daniel)
Put the weights at the path listed in `adapters/__init__.py` (or change the path there) and run
`run.py <name>` then `score.py`. Supported now: any ultralytics YOLO (class names are matched to ours by name; a
`best.json` sidecar's `conf` is used as the operating point), torchvision classifiers saved with
`backbone/head/classes/temperature/threshold` (Brandon's and Daniel's format), Aiden's gate `model.pt`
(loaded through `models/classifier/src/multilabel.py`). Each model is scored at its **own** saved threshold.
