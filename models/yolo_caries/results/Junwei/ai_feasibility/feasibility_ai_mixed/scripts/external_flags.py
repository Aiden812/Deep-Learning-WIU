"""Per-photo flag rates (second half of external_test.py), predicting one photo at a time."""
from pathlib import Path
from ultralytics import YOLO
REPO = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent / "external"
KAGGLE = REPO / "data/raw/kaggle_oral"
CONF = 0.25

def flagged(model, files):
    return sum(len(model.predict(f, conf=CONF, verbose=False)[0].boxes) > 0 for f in files), len(files)

if __name__ == "__main__":
    model = YOLO(REPO / "models/yolo_caries/weights/feasibility_ai_best.pt")
    lab = OUT / "od_caries/labels/test"; img = OUT / "od_caries/images/test"
    for name, keep in (("with caries", True), ("without caries", False)):
        files = [img / f"{p.stem}.jpg" for p in lab.iterdir() if bool(p.read_text().strip()) == keep]
        h, n = flagged(model, files); print(f"OD photos {name}: {h}/{n} flagged ({h / n:.0%})")
    for cls in ("caries", "Normal", "Calculus", "tooth discoloration", "Hypodontia", "Mouth ulcer"):
        files = [p for p in (KAGGLE / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        h, n = flagged(model, files); print(f"Kaggle {cls}: {h}/{n} flagged ({h / n:.0%})")
