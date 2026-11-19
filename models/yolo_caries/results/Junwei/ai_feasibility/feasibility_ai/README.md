# Feasibility run (AI-run, reference only)

Run by Claude (AI) on 2026-11-05 to check whether the Zenodo caries data is learnable before writing the real training notebook. **Not the submitted model.** Logged in the AI Use Log.

## Setup

| | |
|---|---|
| Code | `feasibility_train.py` (same steps as `notebooks/02_train_DRAFT_ai.ipynb`) |
| Data | `data/processed/yolo/` from `scripts/Junwei/dedup_and_split.py`, split by person (`manifests/yolo_split.csv`) |
| Split | train 1,631 images / 293 people, val 270 / 63, test 310 / 63 |
| Model | `yolo11n.pt` (COCO pre-trained), fine-tuned on caries |
| Settings | imgsz 640, batch 16, lr0 0.01, up to 100 epochs, patience 20, workers 4 |
| Hardware | RTX 4050 Laptop (6 GB), 25 min; stopped early at epoch 81, best epoch 61 |
| Weights | `weights/feasibility_ai_best.pt` (gitignored, local only) |

## Results (test split, 310 images from 63 unseen people)

| | Leaky split (superseded) | **Person split (current)** |
|---|---|---|
| mAP50 | 0.944 | **0.924** |
| mAP50-95 | 0.746 | **0.723** |
| Precision | 0.855 | **0.878** |
| Recall | 0.913 | **0.865** |

## The leak and its fix

The first split grouped by patient ID. Zenodo gives one person a new, neighbouring ID per session (with / without cheek retractors), photographed minutes apart, e.g. `003-103-297-01` and `003-103-298-00`. 293 such pairs were split across train/val/test. `evidence/same_person_pairs.jpg` shows three pairs: same mouths.

`person_groups()` in `dedup_and_split.py` now links IDs with opposite sessions, ID numbers <= 2 apart and photos <= 30 min apart, and the split is by person group. Leak check after the fix: 0 same-person pairs across splits. Residual risk: 37 IDs found no partner; about a dozen could be missed pairs (> 30 min apart or 3+ numbers off).

The leak inflated test mAP50 by only ~0.02, so the high score is mostly real.

## Caveats

- Test is one dataset (Zenodo): same clinics, cameras and annotators as training. Expect lower scores on users' phone photos. Not yet tested on external data (Kaggle oral, DentalAI).
- 63 test people: treat scores as roughly +/- 0.02. One split, one seed.
- Recall 0.865: misses about 1 in 7 annotated lesions.

## Files

| Folder | Contents |
|---|---|
| `train/` | per-epoch metrics (`results.csv`, `results.png`), run settings (`args.yaml`), label statistics (`labels.jpg`) |
| `test/` | precision-recall and F1 curves, confusion matrix, test predictions next to annotator labels |
| `leaky_run_superseded/` | metrics of the first (leaky) run, for comparison |
| `evidence/` | same-person pairs (leak), samples of the 4,038 unannotated images, annotator reference crops |

## The 4,038 unannotated Zenodo images

Not healthy negatives: 1,000 of the 1,001 people who have both annotated and unannotated photos have caries boxes in their annotated ones; annotation covered ~90% of biting-surface views and only ~5–10% of front/side views; images checked caries-free got an empty label file, while these got none. Samples in `evidence/` show some with lesion-like spots. Keep them out of training.
