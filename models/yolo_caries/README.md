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
