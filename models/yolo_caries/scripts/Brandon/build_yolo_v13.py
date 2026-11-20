"""Build the v1.3 YOLO training set: v1.2 pool + the external TRAIN parts of the benchmark manifest.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/models/yolo_caries/scripts/Brandon/build_yolo_v13.py

What changes from v1.2 (see the benchmark report for why):
  * AlphaDent train part: dentist caries outlines -> lesion-sized Caries boxes (labels already made by build_manifest).
  * MetaDent DS3 train part: dentist contours -> boxes. Each contour has a free-text description; it becomes a box of
    one of our classes only if the description names it AND the photo's dentist code agrees (C1 caries, C6 plaque /
    calculus, C7 discoloration, C8 missing teeth). Photos with a code but no matching contour are dropped (an
    unboxed finding would teach "this is background"). Photos with no in-scope code are kept with no boxes.
  * Autooral train part: ulcer boxes from the dentist masks.  SMART-OM normal mucosa: no boxes (hard negatives).
  * One caries box style: v1.2 photos from Zenodo and Roboflow whose caries boxes cover the whole tooth (median box
    2.2% / 1.0% of the photo vs 0.2% for AlphaDent lesions) are left out; DentalAI / Kaggle / external_clean caries
    boxes are lesion-sized and stay.
  * External "fit" photos are written twice (file copies, since Ultralytics merges duplicate paths) so they are
    roughly a third of each epoch.
External photos are shrunk to 1280 px on the long side (training runs at 960) so loading them is fast.
Output: datasets/yolo_work_v1.3/ (outside the repo).
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))      # Deep-Learning-WIU/models
from v13_split import BENCH_OUT, POOL, RAW, WORKSPACE, external_train_rows  # noqa: E402

WORK = WORKSPACE / "datasets" / "yolo_work_v1.3"
IMG, LAB = WORK / "images", WORK / "labels"
NAMES = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]
MAX_SIDE = 1280
EXTERNAL_REPEAT = 2
V12_VAL_SAMPLE = 1200
SEED = 42
WHOLE_TOOTH_CARIES = {"zenodo", "roboflow_oral"}

# MetaDent code -> (our class id, words in the contour description that name it)
MD_RULES = [
    ("C1", 1, re.compile(r"cari|decay")),
    ("C6", 0, re.compile(r"calculus|tartar|plaque|deposit")),
    ("C7", 4, re.compile(r"pigment|stain|discolo|yellow|brown|black(?! triangle)|white spot|chalky|fluorosis|dark")),
    ("C8", 2, re.compile(r"missing|absent|edentul|tooth loss|lost|loss of (the )?(tooth|teeth)|extract")),
]


def shrink_copy(src: Path, dst: Path):
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(dst, quality=92)


def metadent_boxes(z, names, row):
    """-> list of (cls, cx, cy, w, h) or None when the photo can't be boxed reliably."""
    k = row.id.split("/")[1]
    codes = set(str(row.codes).split()) if isinstance(row.codes, str) else set()
    wanted = {c: cls for c, cls, _ in MD_RULES if c in codes}
    f = f"metadent/meta_en/{k}/label.json"
    if not wanted:
        return []                                       # no in-scope finding: a clean negative
    if f not in names:
        return None
    info = json.loads(z.read(f"metadent/meta_en/{k}/info.json"))
    if (info["width"], info["height"]) != (row.width, row.height) and \
            abs(info["width"] / info["height"] - row.width / row.height) > 0.02:
        return None                                     # rotated relative to the saved photo: skip
    boxes, found = [], set()
    for it in json.loads(z.read(f))["items"]:
        d = it.get("description", "").lower()
        for code, cls, rx in MD_RULES:
            if code in wanted and rx.search(d):
                pts = np.array([p for poly in it.get("contours", []) for p in poly], float).reshape(-1, 2)
                if len(pts) < 2:
                    continue
                x1, y1 = np.clip(pts.min(0), 0, 1)
                x2, y2 = np.clip(pts.max(0), 0, 1)
                if x2 - x1 < 0.005 or y2 - y1 < 0.005:
                    continue
                boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                found.add(code)
                break
    if set(wanted) - found:
        return None                                     # a finding without a box: leave the photo out
    return boxes


def write_label(p: Path, boxes):
    p.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))


def main():
    for d in (IMG, LAB):
        d.mkdir(parents=True, exist_ok=True)
    ext = external_train_rows()
    ext = ext[ext.reason.isna() | (ext.reason.astype(str) == "") | (ext.reason.astype(str) == "nan")]
    z = zipfile.ZipFile(RAW / "external/MetaDent/labels/metadent_labels.zip")
    names = set(z.namelist())

    rows, dropped = [], {}
    for r in ext.itertuples():
        if r.dataset == "metadent":
            if pd.isna(r.problem) and r.out_of_scope_only != 1:
                dropped["metadent: no label"] = dropped.get("metadent: no label", 0) + 1
                continue
            boxes = metadent_boxes(z, names, r)
            if boxes is None:
                dropped["metadent: finding without a matching contour"] = dropped.get("metadent: finding without a matching contour", 0) + 1
                continue
        elif r.dataset == "smartom":
            boxes = []
        else:   # alphadent, autooral: labels made by build_manifest
            lab = BENCH_OUT / r.label
            boxes = [tuple([int(v.split()[0]), *map(float, v.split()[1:])]) for v in lab.read_text().splitlines() if v.strip()] \
                if lab.exists() else []
        stem = r.id.replace("/", "__")
        copies = EXTERNAL_REPEAT if r.role13 == "fit" else 1
        for i in range(copies):
            s = stem if i == 0 else f"{stem}__rep{i}"
            shrink_copy(BENCH_OUT / r.image, IMG / f"{s}.jpg")
            write_label(LAB / f"{s}.txt", boxes)
            rows.append(dict(path=(IMG / f"{s}.jpg").as_posix(), dataset=r.dataset, role13=r.role13, rep=i,
                             n_boxes=len(boxes), **{n: int(any(b[0] == j for b in boxes)) for j, n in enumerate(NAMES)}))
    ext_df = pd.DataFrame(rows)

    # ---- v1.2 pool ----------------------------------------------------------------------------------------------
    pool = pd.read_csv(POOL / "pool.csv")
    pool["path"] = [(POOL / p).resolve().as_posix() for p in pool.image]

    def has_caries_box(lbl):
        f = POOL / lbl
        return f.exists() and any(l.split()[:1] == ["1"] for l in f.read_text().splitlines())

    pool["whole_tooth_caries"] = pool.dataset.isin(WHOLE_TOOTH_CARIES) & pool.label.map(has_caries_box)
    by = pool.set_index("path")

    def read_list(name):
        return [(POOL / l.strip()).resolve().as_posix() for l in (POOL / name).read_text().splitlines() if l.strip()]

    v12_train = [p for p in read_list("yolo_train.txt") if not by.at[p, "whole_tooth_caries"]]
    v12_val_all = [p for p in read_list("val.txt") if not by.at[p, "whole_tooth_caries"]]
    v12_val = list(pd.Series(v12_val_all).sample(min(V12_VAL_SAMPLE, len(v12_val_all)), random_state=SEED))
    dropped["v1.2 train: whole-tooth caries photos (zenodo/roboflow)"] = len(read_list("yolo_train.txt")) - len(v12_train)

    train = v12_train + ext_df[ext_df.role13 == "fit"].path.tolist()
    val = v12_val + ext_df[ext_df.role13 == "val"].path.tolist()
    (WORK / "train.txt").write_text("\n".join(train) + "\n")
    (WORK / "val.txt").write_text("\n".join(val) + "\n")
    (WORK / "val_external.txt").write_text("\n".join(ext_df[ext_df.role13 == "val"].path) + "\n")
    yaml = lambda tr, va: f"train: {tr.as_posix()}\nval: {va.as_posix()}\nnc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n"
    (WORK / "data.yaml").write_text(yaml(WORK / "train.txt", WORK / "val.txt"))
    (WORK / "eval_val_external.yaml").write_text(yaml(WORK / "train.txt", WORK / "val_external.txt"))
    old = WORKSPACE / "datasets" / "yolo_work_v1.2"
    for n in ("test_fair_human", "test_fair", "test"):
        f = old / f"Y2_yolo11n_640_v12_{n}.txt"
        if f.exists():
            (WORK / f"eval_v12_{n}.yaml").write_text(yaml(WORK / "train.txt", f))
    ext_df.to_csv(WORK / "external_rows.csv", index=False)

    summary = dict(
        train_photos=len(train), train_from_v12=len(v12_train), train_external_incl_repeats=len(train) - len(v12_train),
        val_photos=len(val), val_from_v12=len(v12_val), val_external=int((ext_df.role13 == "val").sum()),
        external_by_dataset=ext_df[ext_df.rep == 0].groupby(["dataset", "role13"]).size().unstack(fill_value=0).to_dict(),
        external_boxes_by_class=ext_df[ext_df.rep == 0].groupby("role13")[NAMES].sum().to_dict(),
        dropped=dropped)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
