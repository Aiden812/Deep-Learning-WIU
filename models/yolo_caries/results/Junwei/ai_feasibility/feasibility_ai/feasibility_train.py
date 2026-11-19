"""Throwaway feasibility run: is the Zenodo split learnable? Not the learner's submission."""

from pathlib import Path

from ultralytics import YOLO

DATA = Path(__file__).resolve().parents[4] / "models/yolo_caries/configs/data.yaml"
OUT = Path(__file__).parent

if __name__ == "__main__":  # Windows needs this guard for multi-process data loading
    YOLO("yolo11n.pt").train(data=DATA, epochs=100, patience=20, imgsz=640, workers=4, project=OUT, name="feasibility_v2", exist_ok=True)
    best = OUT / "feasibility_v2" / "weights" / "best.pt"
    m = YOLO(best).val(data=DATA, split="test", project=OUT, name="feasibility_v2_test", exist_ok=True)
    print(f"TEST mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} precision={m.box.mp:.3f} recall={m.box.mr:.3f}")
