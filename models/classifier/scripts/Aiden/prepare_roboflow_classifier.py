"""Build the 6-class classifier dataset from the Roboflow "oral-diseases" v2 export.

Usage (from the repo root):
    python models/classifier/scripts/Aiden/prepare_roboflow_classifier.py data/raw/roboflow_oral/oral-diseases.v2i.yolov11.zip data/processed/classifier_roboflow

The Roboflow export is a detection dataset (boxes), and a third of its photos show several conditions.
Each photo's class is taken from its filename prefix instead (e.g. "caries-1004-_jpg..."), i.e. the folder
the photo came from in the original dataset.
Roboflow's own train/valid/test split is kept: valid and test are un-augmented, while train holds 3 augmented
copies of every photo. All copies share one `group`, and the notebook keeps a group inside one split.
The dataset has no Normal (healthy) class.
Writes <out>/<split folders>/images/*.jpg and <out>/manifest.csv (file,label,split,group).
"""
import re, sys, zipfile
from pathlib import Path

import pandas as pd

CLASS = {"calculus": "Calculus", "caries": "Caries", "gingivitis": "Gingivitis", "hypodontia": "Hypodontia",
         "tooth_discolation": "Tooth_Discoloration", "ulcer": "Mouth_Ulcer"}   # "discolation" is Roboflow's spelling
SPLIT = {"train": "train", "valid": "val", "test": "test"}


def main(zip_path, out):
    zip_path, out = Path(zip_path), Path(out)
    with zipfile.ZipFile(zip_path) as z:
        images = [n for n in z.namelist() if "/images/" in n and not n.endswith("/")]
        z.extractall(out, members=images)

    rows = []
    for f in images:
        photo = re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", f.split("/")[-1])   # Roboflow appends .rf.<hash> to each copy
        prefix = re.match(r"([a-z_]+?)[-_ (]*\d", photo.lower())
        assert prefix and prefix.group(1) in CLASS, f"no known class at the start of {f}"
        rows.append(dict(file=f, label=CLASS[prefix.group(1)], split=SPLIT[f.split("/")[0]], group=photo))
    df = pd.DataFrame(rows)

    leak = (df.split != "train") & df.group.isin(df[df.split == "train"].group)
    print(f"dropping {leak.sum()} val/test images whose photo is also in train")
    df = df[~leak]

    assert not set(df[df.split == "train"].group) & set(df[df.split != "train"].group), "a photo is in train and val/test"
    df.to_csv(out / "manifest.csv", index=False)
    print(pd.crosstab(df.label, df.split, margins=True))
    print(f"unique photos: {df.group.nunique()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
