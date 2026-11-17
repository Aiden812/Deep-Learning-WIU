"""Score the Zenodo-trained feasibility model on external data (oral-diseases test split, Kaggle oral)."""
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[5]
WEIGHTS = REPO / "models/yolo_caries/weights/feasibility_ai_best.pt"
OD = REPO / "data/raw/oral-diseases.v2i.yolov11/test"
KAGGLE = REPO / "data/raw/kaggle_oral"
OUT = Path(__file__).parent / "external"
CONF = 0.25  # Ultralytics' default prediction threshold


def caries_only_copy() -> Path:
    """oral-diseases test with only caries boxes (class 1 -> 0); photos without caries stay as background."""
    root = OUT / "od_caries"
    for kind in ("images", "labels"):
        (root / kind / "test").mkdir(parents=True, exist_ok=True)
    for img in (OD / "images").iterdir():
        shutil.copy2(img, root / "images/test" / img.name)
        lines = [l.split() for l in (OD / "labels" / f"{img.stem}.txt").read_text().splitlines() if l.strip()]
        keep = [f"0 {' '.join(p[1:5])}\n" for p in lines if p[0] == "1"]
        (root / "labels/test" / f"{img.stem}.txt").write_text("".join(keep))
    yaml = OUT / "od_caries.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: caries\n")
    return yaml


def flagged_share(model, folder: Path) -> tuple[int, int]:
    """How many photos in a folder get at least one caries box at CONF."""
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    hits = sum(len(r.boxes) > 0 for r in model.predict(files, conf=CONF, stream=True, verbose=False))
    return hits, len(files)


if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    m = model.val(data=caries_only_copy(), split="test", workers=2, project=OUT, name="od_test", exist_ok=True)
    print(f"OD box-level: mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} precision={m.box.mp:.3f} recall={m.box.mr:.3f}")

    od_lab = OUT / "od_caries/labels/test"
    with_c = [OUT / "od_caries/images/test" / f"{p.stem}.jpg" for p in od_lab.iterdir() if p.read_text().strip()]
    without = [OUT / "od_caries/images/test" / f"{p.stem}.jpg" for p in od_lab.iterdir() if not p.read_text().strip()]
    for name, files in (("with caries", with_c), ("without caries", without)):
        hits = sum(len(r.boxes) > 0 for r in model.predict(files, conf=CONF, stream=True, verbose=False))
        print(f"OD photos {name}: {hits}/{len(files)} flagged ({hits / len(files):.0%})")

    for cls in ("caries", "Normal", "Calculus", "tooth discoloration"):
        hits, n = flagged_share(model, KAGGLE / cls)
        print(f"Kaggle {cls}: {hits}/{n} flagged ({hits / n:.0%})")
