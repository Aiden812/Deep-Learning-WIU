# Condition classifier

Multi-class image classifier over the classes in `shared/conditions.yaml`.

- Data: `data/processed/classifier/` (splits in `manifests/`)
- Training: `notebooks/`
- Inference: `src/predict.py`
- Weights: `weights/` (gitignored)

## Aiden: stage-1 gate, DenseNet-121 vs DINOv2 ViT-B/14, on Daniel's `oral_dataset`

Binary: healthy or has an issue (`has_issue`). Flagged photos go to the YOLO detector (which condition, where:
`models/yolo_caries/README.md`). The threshold is set to catch 95% of issues on the clean val set. Decisions:
`docs/decisions.md` (2026-11-11, 2026-11-12).

**Data.** Unzip Daniel's `oral_dataset.zip` into `data/raw/oral_dataset/`. About 90% of his val/test images with an
issue are copies of training photos, so `find_leaks.py` marks them and the threshold and headline scores use only the
**clean** val/test images (his full test is reported alongside). `find_leaks.py` reuses the copy-matching cache in
`data/processed/classifier_multilabel/_cache/` (built once by `prepare_roboflow_multilabel.py`).

```bash
# environment (CUDA build of PyTorch first)
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r models/classifier/notebooks/Aiden/requirements.txt

python models/classifier/scripts/Aiden/find_leaks.py      # once per dataset version: marks val/test copies of training photos
python models/classifier/scripts/Aiden/train.py --arch densenet121 --lr 1e-4
python models/classifier/scripts/Aiden/train.py --arch dinov2_vitb14
```

Each command overwrites `results/Aiden/<arch>/` (`model.pt`, `metrics.json`, `test_report.txt`, `train_log.txt`,
`val_predictions.csv`, `test_predictions.csv`) and regenerates the single `results/Aiden/comparison_report.txt` and
the graphs `results/Aiden/figures/comparison_*.png` (`scripts/Aiden/report.py`; run it alone to rebuild them).
Then open `notebooks/Aiden/01_gate_classifier.ipynb` (`.venv` kernel, Run All) for the walkthrough: data, leakage,
the source problem, results per source, most confident mistakes, Grad-CAM, experiment log. It reads the saved results;
set `RETRAIN = True` in it to retrain both models.

`model.pt` holds `classes=["has_issue"]`, the calibration (`calib_a`, `calib_b`) and the `thresholds`;
`src/multilabel.py` (`load_model`, `predict`) loads it.

**Known limitation.** In the clean test every image with an issue comes from Roboflow and every healthy image from another
source, so it can't separate "recognises disease" from "recognises Roboflow-style photos" (notebook section 4).

| File | Purpose |
|---|---|
| `notebooks/Aiden/01_gate_classifier.ipynb` | walkthrough and checks of the gate (reuses `train.py` / `report.py`) |
| `notebooks/Aiden/00_old_roboflow_multilabel.ipynb` | earlier approach, kept for the record: 6-condition multi-label on the Roboflow export only |
| `scripts/Aiden/train.py`, `report.py` | train one gate model; comparison report + graphs |
| `scripts/Aiden/find_leaks.py` | marks val/test images of `oral_dataset` that copy a training photo |
| `scripts/Aiden/prepare_roboflow_multilabel.py` | builds the copy-matching cache `find_leaks.py` needs (and the old multi-label dataset) |
| `scripts/Aiden/filter_images.py` | Quality check for a folder of images (tiny / dark / blurry / greyscale / duplicate flags + review sheets); used on Daniel's scraped images |
