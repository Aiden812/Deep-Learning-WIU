# Condition classifier

Multi-class image classifier over the classes in `shared/conditions.yaml`.

- Data: `data/processed/classifier/` (splits in `manifests/`)
- Training: `notebooks/`
- Inference: `src/predict.py`
- Weights: `weights/` (gitignored)

## Aiden: DenseNet-121 on the Roboflow oral-diseases dataset

6 classes: Calculus, Caries, Gingivitis, Hypodontia, Mouth_Ulcer, Tooth_Discoloration. **No Normal class.**

**Data.** Roboflow *oral-diseases* v2 (see the main README). 9,155 photos: 7,324 train (each stored as 3 augmented
copies), 916 val, 915 test. It is a detection export, so each photo's class comes from its filename prefix.

```bash
# 1. put the Roboflow zip in data/raw/roboflow_oral/, then build data/processed/classifier_roboflow/
python models/classifier/scripts/Aiden/prepare_roboflow_classifier.py data/raw/roboflow_oral/oral-diseases.v2i.yolov11.zip data/processed/classifier_roboflow

# 2. environment (CUDA build of PyTorch first)
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scipy scikit-learn matplotlib pillow ipykernel
```

3. Open `notebooks/Aiden/01_condition_classifier.ipynb`, pick the `.venv` kernel, Run All. It also runs on Kaggle
   (upload `classifier_roboflow` as a dataset). Outputs go to `results/Aiden/`; the `.pt` files there are gitignored.

| File | Purpose |
|---|---|
| `notebooks/Aiden/01_condition_classifier.ipynb` | Near-duplicate check, training, test metrics, Grad-CAM, experiment log |
| `scripts/Aiden/prepare_roboflow_classifier.py` | Roboflow zip → `classifier_roboflow/` + `manifest.csv` (file, label, split, group) |
| `scripts/Aiden/filter_images.py`, `prepare_classifier_data.py` | Run 1 pipeline: quality filter + cleaning for the Kaggle oral dataset |
