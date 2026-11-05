# Condition classifier

Multi-class image classifier over the classes in `shared/conditions.yaml`.

- Data: `data/processed/classifier/` (splits in `manifests/`)
- Training: `notebooks/`
- Inference: `src/predict.py`
- Weights: `weights/` (gitignored)

## Aiden: DenseNet-121, multi-label, on the Roboflow oral-diseases dataset

Answers yes/no for each of 6 conditions: Calculus, Caries, Gingivitis, Hypodontia, Mouth_Ulcer, Tooth_Discoloration.
A photo can have several. **No Normal class.** Why multi-label and why the data is re-split: `docs/decisions.md` (2026-11-05).

**Data.** Roboflow *oral-diseases* v2 (see the main README). Its own split leaks copies of training photos into test, so
the prep script groups every copy of an original photo (DINOv2 candidates confirmed by SIFT keypoint matching) and splits
70/15/15 by group; val/test keep one image per split group (no copies of each other). Labels = box classes + folder class, unioned over copies.

```bash
# 1. environment (CUDA build of PyTorch first)
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r models/classifier/notebooks/Aiden/requirements.txt

# 2. put the Roboflow zip in data/raw/roboflow_oral/, then build data/processed/classifier_multilabel/
#    (needs a CUDA GPU; about an hour the first time, cached in _cache/ afterwards)
python models/classifier/scripts/Aiden/prepare_roboflow_multilabel.py data/raw/roboflow_oral/oral-diseases.v2i.yolov11.zip data/processed/classifier_multilabel
```

3. Open `notebooks/Aiden/01_condition_classifier.ipynb`, pick the `.venv` kernel, Run All. It also runs on Kaggle
   (upload `classifier_multilabel` without `_cache`). Outputs go to `results/Aiden/`; the `.pt` files there are gitignored.

The saved `condition_classifier.pt` holds `classes`, per-condition `thresholds` and `multilabel: True`: a condition is
reported when its sigmoid probability is at or above its threshold.

| File | Purpose |
|---|---|
| `notebooks/Aiden/01_condition_classifier.ipynb` | EDA, leakage checks, training, thresholds, test metrics, Grad-CAM, experiment log |
| `scripts/Aiden/prepare_roboflow_multilabel.py` | Roboflow zip → `classifier_multilabel/` + `manifest.csv` (file, split, group, one 0/1 column per condition) |
| `scripts/Aiden/filter_images.py`, `prepare_classifier_data.py` | Earlier pipeline (no longer used): quality filter + cleaning for the Kaggle oral dataset |
