# YOLO caries detector

Detects carious lesions on intraoral photos.

## Pipeline

1. `scripts/check_downloads.py` — verify `data/raw/` is populated
2. `scripts/convert_dentalai_to_yolo.py` — DentalAI masks → YOLO boxes (`data/interim/`)
3. `scripts/dedup_and_split.py` — dedup across sources, write splits + `manifests/yolo_split.csv`
4. `scripts/visualise_labels.py` — eyeball labels before training
5. Train (notebook in `notebooks/`) with `configs/data.yaml`
6. Log every run in `results/experiments.csv`; best weights go in `weights/` (gitignored)

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
