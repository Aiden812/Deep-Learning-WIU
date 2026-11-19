"""Score a trained caries model on photos it never trained on.

1. Box quality (mAP) on the Zenodo test split and on the oral-diseases test split.
2. Photo-level alarms: the share of photos that get at least one caries box, at several confidence cut-offs, for
   healthy Mendeley test photos and for one part of the Kaggle caries / Normal photos (see build_mixed.py).

Pick the cut-off with --part tune, then report once with --part report.

Usage:
    python models/yolo_caries/scripts/Junwei/eval_mixed.py --weights <path to best.pt> [--part report]
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[4]
CUTOFFS = (0.25, 0.4, 0.5, 0.6)


def top_confidence(model: YOLO, files: list[Path]) -> list[float]:
    """Highest caries confidence in each photo, 0 if no box. One photo at a time: batches ran out of RAM on 15 GB."""
    out = []
    for f in files:
        conf = model.predict(f, conf=min(CUTOFFS), verbose=False)[0].boxes.conf
        out.append(float(conf.max()) if len(conf) else 0.0)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--part", choices=["tune", "report"], default="tune")
    args = parser.parse_args()
    model = YOLO(args.weights)

    for name, data in (("Zenodo test", REPO / "models/yolo_caries/configs/data.yaml"),
                       ("oral-diseases test", REPO / "data/processed/yolo_mixed_od.yaml")):
        m = model.val(data=data, split="test", workers=2, plots=False, verbose=False)
        print(f"RESULT {name}: mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} "
              f"precision={m.box.mp:.3f} recall={m.box.mr:.3f}")

    with (REPO / "manifests/yolo_kaggle_split.csv").open() as f:
        kaggle = [r for r in csv.DictReader(f) if r["part"] == args.part]
    sets = {
        "healthy (Mendeley test)": sorted((REPO / "data/processed/yolo_mixed/images/test").glob("men__*")),
        f"Kaggle Normal ({args.part})": [REPO / r["path"] for r in kaggle if r["label"] == "Normal"],
        f"Kaggle caries ({args.part})": [REPO / r["path"] for r in kaggle if r["label"] == "caries"],
    }
    print("\nShare of photos flagged (at least one caries box). Healthy and Normal should be low, caries high.")
    print("cut-off  " + "  ".join(f"{name} n={len(files)}" for name, files in sets.items()))
    tops = {name: top_confidence(model, files) for name, files in sets.items()}
    for c in CUTOFFS:
        print(f"{c:<8} " + "  ".join(f"{sum(t >= c for t in tops[name]) / len(tops[name]):>{len(name) + 3 + len(str(len(tops[name])))}.0%}"
                                      for name in sets))
