"""Package the external TRAIN part (the one v1.3 trained on) as a standalone dataset the team can copy.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/models/export_external_train_v13.py

Reads what build_yolo_v13.py already made (datasets/yolo_work_v1.3), keeps one copy of each photo (no __rep files),
and writes a portable folder + zip with relative paths:

    datasets/external_train_v13/
        images/{fit,val}/*.jpg     1280 px long side
        labels/{fit,val}/*.txt     YOLO boxes, our 5 class ids (empty file = no boxes)
        data.yaml                  Ultralytics config
        labels.csv                 one row per photo: split, source, patient group, photo-level labels
        README.md
    datasets/external_train_v13.zip

Only manifest role == "train" photos are in it; the bench part stays unseen, so team models can still be scored on it.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v13_split import WORKSPACE, external_train_rows, role13  # noqa: E402

SRC = WORKSPACE / "datasets" / "yolo_work_v1.3"
OUT = WORKSPACE / "datasets" / "external_train_v13"
NAMES = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]
TRUTH = ["calculus", "caries", "missing_teeth", "ulcer", "discoloration", "problem"]

README = """# External training set (v1.3)

The photos from AlphaDent, MetaDent (DS3), Autooral and SMART-OM that the benchmark manifest marks as **train**.
Brandon's YOLO v13 and classifier v13 were trained on exactly these (plus our v1.2 pool).
The benchmark (**bench**) photos are NOT here, so models trained on this can still be scored fairly with `benchmark/`.
Do not add photos from the raw downloads yourself: that would leak bench photos into training.

## Counts

{counts}

## Splits

`fit` = train on it. `val` = pick the best epoch / threshold on it, never train on it.
The split is by patient / near-duplicate group (same as `models/v13_split.py`), so one patient is never in both.

## YOLO (Ultralytics)

Classes: {names} (same ids as `shared/conditions.yaml` / v1.2).
Edit `path:` in `data.yaml` to where you unzipped this folder (absolute path), then:

```python
from ultralytics import YOLO
YOLO("your_start_weights.pt").train(data="C:/.../external_train_v13/data.yaml", imgsz=960)
```

To mix with your own data, list both image folders under `train:` in the yaml.

Box styles: caries boxes are lesion-sized (AlphaDent / MetaDent), not whole-tooth like Zenodo/Roboflow in v1.2.
SMART-OM photos and MetaDent photos with no in-scope finding have empty label files: they are negatives on purpose.
SMART-OM "Normal" means normal mucosa only; teeth in those photos are not checked.

## Classifier

`labels.csv` has photo-level labels from the dentists' codes: `calculus, caries, missing_teeth, ulcer, discoloration,
problem` = 1 present / 0 absent / empty = unknown. `box_*` columns say which YOLO classes have a box in that photo.
AlphaDent photos without caries are NOT confirmed healthy (only caries was outlined), so the v13 classifier left
AlphaDent out; do the same unless you only use it for caries.
"""


def main():
    rows = pd.read_csv(SRC / "external_rows.csv")
    rows = rows[rows.rep == 0].copy()
    rows["stem"] = rows.path.map(lambda p: Path(p).stem)

    man = external_train_rows()
    man["stem"] = man.id.str.replace("/", "__", regex=False)
    df = rows.merge(man[["stem", "id", "group", "subset", "codes", *TRUTH]], on="stem", how="left", validate="1:1")
    assert df.id.notna().all(), "row without a manifest entry"
    assert (df.role13 == df.group.astype(str).map(role13)).all(), "split differs from v13_split"

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("fit", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
    for r in df.itertuples():
        shutil.copy2(r.path, OUT / "images" / r.role13 / f"{r.stem}.jpg")
        shutil.copy2(SRC / "labels" / f"{r.stem}.txt", OUT / "labels" / r.role13 / f"{r.stem}.txt")

    out = pd.DataFrame(dict(
        image=[f"images/{s}/{t}.jpg" for s, t in zip(df.role13, df.stem)],
        label=[f"labels/{s}/{t}.txt" for s, t in zip(df.role13, df.stem)],
        split=df.role13, dataset=df.dataset, subset=df.subset, group=df.group, source_id=df.id, codes=df.codes,
        **{c: df[c].astype("Int64") for c in TRUTH},
        n_boxes=df.n_boxes, **{f"box_{n}": df[n] for n in NAMES}))
    out.sort_values(["split", "dataset", "image"]).to_csv(OUT / "labels.csv", index=False)

    (OUT / "data.yaml").write_text(
        f"# set path to where this folder is on your machine (absolute)\npath: {OUT.as_posix()}\n"
        f"train: images/fit\nval: images/val\nnc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n")

    counts = out.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.loc["total"] = counts.sum()
    boxes = out.groupby("split")[[f"box_{n}" for n in NAMES]].sum().T
    boxes.index = [i[4:] for i in boxes.index]
    to_md = lambda t, first: "| " + " | ".join([first, *map(str, t.columns)]) + " |\n|" + "---|" * (len(t.columns) + 1) + "\n" + \
        "\n".join("| " + " | ".join([str(i), *map(str, v)]) + " |" for i, v in zip(t.index, t.values))
    (OUT / "README.md").write_text(README.format(
        names=", ".join(f"{i} {n}" for i, n in enumerate(NAMES)),
        counts="Photos:\n\n" + to_md(counts, "dataset") + "\n\nPhotos with at least one box of the class:\n\n" + to_md(boxes, "class")),
        encoding="utf-8")

    z = OUT.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:     # jpgs don't compress further
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(OUT.parent).as_posix())
    print(counts.to_string())
    print(boxes.to_string())
    print(f"-> {OUT}\n-> {z} ({z.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
