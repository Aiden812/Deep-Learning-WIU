# YOLO detector

Draws a box around each problem on a mouth photo and names it: Calculus, Caries, Hypodontia, Mouth_Ulcer, Tooth_Discoloration.
All team runs train on Daniel's `data/raw/oral_dataset/` (see its README) and are logged in `results/experiments.csv`.
Weights are gitignored.

## Jun Wei: YOLO11n at 640 vs 960 px

1. `notebooks/JunWei/01_eda.ipynb`: checks the data (Zenodo labels and patient leak, then class counts, boxes per image, box sizes and sources in `oral_dataset`)
2. `notebooks/JunWei/02_training_script.ipynb`: trains YOLO11n at 640 and 960 px with the same settings, tests both on the test set and logs them
3. Results: `results/Junwei/` (see its README)

## Earlier Zenodo caries pipeline (before the team dataset)

`scripts/Junwei/`: `check_downloads.py` (check `data/raw/`), `convert_dentalai_to_yolo.py` (DentalAI masks to YOLO boxes),
`dedup_and_split.py` (split Zenodo by person, tested in `tests/test_dedup_and_split.py`), `build_mixed.py` (add other caries and healthy photos),
`visualise_labels.py` (look at labels). `configs/data.yaml` belongs to this pipeline.

## Inference

```python
from models.yolo_caries.src.predict import predict
```

## Aiden: 5-class detector on Daniel's `oral_dataset` (stage 2 behind the classifier gate)

```bash
pip install ultralytics
python models/classifier/scripts/Aiden/find_leaks.py         # once: marks val/test copies of training photos
python models/yolo_caries/scripts/Aiden/train_yolo.py         # yolo11n, 640 px; --model yolo11s.pt etc. to compare
```

Trains on Daniel's `yolo_train.txt` unchanged. `best.pt` is picked on the val set **without copies of training photos**.
Test is reported on Daniel's `data.yaml` (the number to compare across the team, logged in `results/experiments.csv`),
on `data_person.yaml`, and on the copy-free subset (the honest score: about 90% of Daniel's val/test images with a
condition are copies of training photos). Output: `results/Aiden/<model>/` (overwritten per run), `results/Aiden/yolo_report.txt`
and `results/Aiden/figures/yolo_*.png`.
