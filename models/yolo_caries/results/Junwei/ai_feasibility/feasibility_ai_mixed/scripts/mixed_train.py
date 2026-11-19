"""AI feasibility: train on Zenodo + oral-diseases caries, then score on Zenodo test, oral-diseases test and Kaggle."""
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[5]
MIXED = REPO / "data/processed/yolo_mixed.yaml"
ZEN = REPO / "models/yolo_caries/configs/data.yaml"
OD = Path(__file__).parent / "external/od_caries.yaml"   # oral-diseases test, caries only (from external_test.py)
KAGGLE = REPO / "data/raw/kaggle_oral"
OUT = Path(__file__).parent / "smoke"
CONF = 0.25

if __name__ == "__main__":
    YOLO("yolo11n.pt").train(data=MIXED, epochs=100, patience=20, imgsz=640, workers=2,
                             project=OUT, name="mixed", exist_ok=True)
    model = YOLO(OUT / "mixed/weights/best.pt")
    for tag, data in (("Zenodo test", ZEN), ("oral-diseases test", OD)):
        m = model.val(data=data, split="test", workers=2, project=OUT, name=f"mixed_test_{tag.split()[0]}", exist_ok=True)
        print(f"RESULT {tag}: mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} precision={m.box.mp:.3f} recall={m.box.mr:.3f}")
    for cls in ("caries", "Normal", "Calculus", "tooth discoloration", "Hypodontia", "Mouth ulcer"):
        files = [p for p in (KAGGLE / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        hits = sum(len(model.predict(f, conf=CONF, verbose=False)[0].boxes) > 0 for f in files)
        print(f"RESULT Kaggle {cls}: {hits}/{len(files)} flagged ({hits / len(files):.0%})")
