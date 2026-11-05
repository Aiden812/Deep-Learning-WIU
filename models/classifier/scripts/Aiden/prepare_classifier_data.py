"""Build the cleaned 6-class dataset for the condition classifier.

Usage (from the repo root):
    python models/classifier/scripts/Aiden/prepare_classifier_data.py data/interim/filter_kaggle/report.csv data/processed/classifier

Drops what filter_images.py flagged as unusable, plus anything listed in
data/interim/rejects.csv (columns: sheet,tile,reason) from the manual review.
Every kept image is shrunk so its long side is at most MAX_SIDE, because the
Normal photos are ~6x larger than the rest and the model must not learn
"large sharp photo = Normal".
Writes <out>/<Class>/*.jpg and <out>/manifest.csv.
"""
import csv, sys
from pathlib import Path
import pandas as pd
from PIL import Image

MAX_SIDE = 320  # close to the median size of the non-Normal classes
DROP = "corrupt|too_small|duplicate|label_conflict|greyscale"  # blur/brightness are reviewed by eye, not auto-dropped
CLASS_NAMES = {"Calculus": "Calculus", "caries": "Caries", "Hypodontia": "Hypodontia",
               "Mouth ulcer": "Mouth_Ulcer", "Normal": "Normal", "tooth discoloration": "Tooth_Discoloration"}


def main(report_csv, out):
    report_csv, out = Path(report_csv), Path(out)
    rep = pd.read_csv(report_csv)
    flags = rep["flags"].fillna("")
    keep = rep[~flags.str.contains(DROP)]
    rejects = report_csv.parent.parent / "rejects.csv"
    if rejects.exists():
        rej = pd.read_csv(rejects)
        bad = set(zip(rej["sheet"], rej["tile"]))
        keep = keep[[(s, t) not in bad for s, t in zip(keep["sheet"], keep["tile"])]]
        print(f"applied {len(bad)} manual rejects from {rejects}")
    else:
        print(f"no {rejects} yet: manual review not applied")

    rows = []
    for i, r in enumerate(keep.to_dict("records")):
        cls = CLASS_NAMES[r["class"]]
        dst = out / cls / f"{cls}_{i:05d}.jpg"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(r["path"]).convert("RGB")  # report paths are relative to the Project folder
            im.thumbnail((MAX_SIDE, MAX_SIDE))  # only ever shrinks
            im.save(dst, quality=90)
        rows.append({"file": f"{cls}/{dst.name}", "label": cls, "source": r["path"]})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "source"])
        w.writeheader()
        w.writerows(rows)

    before = rep["class"].map(CLASS_NAMES).value_counts()
    after = pd.Series([r["label"] for r in rows]).value_counts()
    print(pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
