"""Build data/processed/yolo_mixed: the Zenodo split + oral-diseases caries + healthy mouths, all as one class.

Zenodo         data/processed/yolo as written by dedup_and_split.py (run that first), unchanged.
oral-diseases  Caries boxes only (class 1 -> 0). Train keeps one copy per source photo (Roboflow made 3) and adds
               OD_BACKGROUND photos without caries, only from discoloration/ulcer sources (no cross-split copies).
               Its valid/test keep every photo.
Mendeley       Photos of teeth without caries, added with an empty label so the model learns to draw nothing.
               The same number from each of the 8 views; look-alike photos are skipped.
Kaggle         The brief's core dataset. It has no boxes, so only its Normal photos can be trained on: a third go to
               train with an empty label. The rest, and all caries photos, are kept for 'tune' (picking the confidence
               cut-off) and 'report' (the final check). manifests/yolo_kaggle_split.csv says which photo is which.

Usage:
    python models/yolo_caries/scripts/Junwei/build_mixed.py
"""
import csv
import random
import re
import shutil
from pathlib import Path

import imagehash
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
ZEN = REPO / "data/processed/yolo"
OD = REPO / "data/raw/oral-diseases.v2i.yolov11"
MENDELEY = REPO / "data/raw/mendeley_noncaries"
KAGGLE = REPO / "data/raw/kaggle_oral"
DST = REPO / "data/processed/yolo_mixed"
KAGGLE_MANIFEST = REPO / "manifests/yolo_kaggle_split.csv"
# Shuffled photos are dealt out in this order. Caries photos have no boxes, so none can go to train.
KAGGLE_PARTS = {"caries": ("report", "tune"), "Normal": ("report", "tune", "report", "tune", "train", "train")}
OD_BACKGROUND = 350  # no-caries photos from oral-diseases in train
HEALTHY_PER_VIEW = {"train": 75, "val": 15, "test": 50}  # x 8 Mendeley views = 600 / 120 / 400 healthy photos
NEAR_BITS = 16  # perceptual hashes this close are usually the same mouth: a re-save, a crop or the next burst shot
CARIES = "1"


def source_of(stem: str) -> str:
    return re.sub(r"\.rf\.[0-9a-f]+$", "", stem)


def caries_lines(label: Path) -> list[str]:
    rows = [l.split() for l in label.read_text().splitlines() if l.strip()]
    return [f"0 {' '.join(r[1:5])}\n" for r in rows if r[0] == CARIES]


def distinct(files: list[Path], taken: list, want: int | None = None) -> list[Path]:
    """Walk files in order and keep those that do not look like a photo already kept (their hashes are in `taken`)."""
    kept = []
    for f in files:
        if len(kept) == want:
            break
        h = imagehash.phash(Image.open(f))
        if all(h - t > NEAR_BITS for t in taken):
            taken.append(h)
            kept.append(f)
    return kept


def healthy_by_split(rng: random.Random) -> dict[str, list[Path]]:
    """Mendeley no-caries photos for each split. No photo looks like another one, within or across splits.

    shortcut: Mendeley has no person IDs, so one person can still be in two splits in different views;
    treat its test photos as a rough check. A person-clean healthy test needs a dataset with IDs.
    """
    out, taken, need = {s: [] for s in HEALTHY_PER_VIEW}, [], sum(HEALTHY_PER_VIEW.values())
    for view in sorted(p for p in MENDELEY.iterdir() if p.is_dir()):
        files = sorted(view.glob("*.jpg"))
        rng.shuffle(files)
        picked = distinct(files, taken, need)
        assert len(picked) == need, f"only {len(picked)} distinct photos in {view.name}, need {need}"
        for s, n in HEALTHY_PER_VIEW.items():
            out[s] += picked[:n]
            picked = picked[n:]
    return out


def write_kaggle_parts(rng: random.Random) -> list[Path]:
    """Deal Kaggle caries and Normal photos into train/tune/report and return the train photos.

    Look-alikes are dropped first, so no copy of a photo sits in two parts.
    """
    rows = []
    for label in ("caries", "Normal"):
        files = sorted(p for p in (KAGGLE / label).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        files = distinct(files, [])
        rng.shuffle(files)
        parts = KAGGLE_PARTS[label]
        rows += [{"path": f.relative_to(REPO).as_posix(), "label": label, "part": parts[i % len(parts)]}
                 for i, f in enumerate(files)]
        print(f"Kaggle {label}: {len(files)} distinct photos -> "
              + ", ".join(f"{p} {sum(r['label'] == label and r['part'] == p for r in rows)}" for p in sorted(set(parts))))
    with KAGGLE_MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "part"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["path"]))
    return [REPO / r["path"] for r in rows if r["part"] == "train"]


if __name__ == "__main__":
    rng = random.Random(42)
    healthy = healthy_by_split(rng)
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        img_dir, lab_dir = DST / "images" / split, DST / "labels" / split
        img_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)
        zen = list((ZEN / "images" / split).glob("*.jpg"))
        for img in zen:  # Zenodo, unchanged
            shutil.copy2(img, img_dir / f"zen__{img.name}")
            shutil.copy2(ZEN / "labels" / split / f"{img.stem}.txt", lab_dir / f"zen__{img.stem}.txt")

        od_split = {"train": "train", "val": "valid", "test": "test"}[split]
        imgs = sorted((OD / od_split / "images").iterdir())
        if split == "train":
            first = {}
            for p in imgs:
                first.setdefault(source_of(p.stem), p)  # one copy per source photo
            imgs = list(first.values())
        with_c, without = [], []
        for p in imgs:
            lines = caries_lines(OD / od_split / "labels" / f"{p.stem}.txt")
            (with_c if lines else without).append((p, lines))
        if split == "train":
            pool = [x for x in without if source_of(x[0].stem).lower().startswith(("tooth_discolation", "ulcer"))]
            without = rng.sample(pool, min(OD_BACKGROUND, len(pool)))
        for p, lines in with_c + without:
            shutil.copy2(p, img_dir / f"od__{p.name}")
            (lab_dir / f"od__{p.stem}.txt").write_text("".join(lines))

        for p in healthy[split]:
            name = f"men__{p.parent.name}__{p.stem}".replace(" ", "_")
            shutil.copy2(p, img_dir / f"{name}.jpg")
            (lab_dir / f"{name}.txt").write_text("")
        print(f"{split}: {sum(1 for _ in img_dir.iterdir())} photos = Zenodo {len(zen)} + oral-diseases "
              f"{len(with_c)} with caries and {len(without)} without + healthy {len(healthy[split])}")

    # oral-diseases test on its own, so it can be scored apart from Zenodo
    (DST / "test_od.txt").write_text("".join(f"./images/test/{p.name}\n" for p in sorted((DST / "images/test").glob("od__*"))))
    for name, test in (("yolo_mixed", "images/test"), ("yolo_mixed_od", "test_od.txt")):
        (DST.parent / f"{name}.yaml").write_text(
            f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\ntest: {test}\nnames:\n  0: caries\n")
    for p in write_kaggle_parts(rng):  # healthy phone-style photos; Kaggle "Normal" is not dentist-checked
        shutil.copy2(p, DST / "images/train" / f"kag__Normal__{p.name}")
        (DST / "labels/train" / f"kag__Normal__{p.stem}.txt").write_text("")
    print(f"train with Kaggle Normal: {sum(1 for _ in (DST / 'images/train').iterdir())} photos")
