"""Step 3: build the frozen benchmark: which photos, which labels, bench part vs train part.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/build_manifest.py

Reads the raw datasets in data/raw (AlphaDent, MetaDent + its labels zip, Autooral, SMART-OM) and the two copy checks
(phash_matches.csv, sift_matches.csv). Writes:
  benchmark/manifest.csv                 one row per photo (committed; this IS the benchmark)
  data/processed/benchmark/images/...    the photos, longest side <= 1600 px, EXIF rotation applied, JPEG q95
  data/processed/benchmark/labels/...    YOLO boxes with our 5 class ids (AlphaDent caries, Autooral ulcers)

role = bench     scored by the benchmark; nobody may ever train on these
       train     may be used for training later (never scored)
       excluded  copy of a team training photo, or not usable (reason column says why)

Photo-level truth columns (calculus, caries, missing_teeth, ulcer, discoloration, problem): 1 / 0 / empty = unknown.
"""
import glob
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COPY_DIST, MANIFEST, OUT, RAW, SEED, SIFT_COPY  # noqa: E402

MAX_SIDE = 1600
IMG, LAB = OUT / "images", OUT / "labels"
rng = np.random.default_rng(SEED)
# MetaDent codes (classification.json) -> our photo-level columns
MD_MAP = {"C1": "caries", "C6": "calculus", "C7": "discoloration", "C8": "missing_teeth", "C17": "ulcer"}
MD_PROBLEM_OOS = {"C2", "C3", "C4", "C5", "C9", "C18"}   # real problems our 5 classes don't cover (gums, wear, ...)
COLS = ["calculus", "caries", "missing_teeth", "ulcer", "discoloration", "problem"]

# PIL exif_transpose operation for each EXIF orientation, and the same move for normalised (x, y) points
ORIENT_PTS = {2: lambda x, y: (1 - x, y), 3: lambda x, y: (1 - x, 1 - y), 4: lambda x, y: (x, 1 - y),
              5: lambda x, y: (y, x), 6: lambda x, y: (1 - y, x), 7: lambda x, y: (1 - y, 1 - x), 8: lambda x, y: (y, 1 - x)}


def save_img(im, dst):
    """EXIF-rotate, shrink to MAX_SIDE, save. Returns (orientation, w, h) of the saved photo."""
    orient = im.getexif().get(274, 1)
    if dst.exists():                      # already cached by an earlier run
        with Image.open(dst) as done:
            return orient, done.size
    if im.format == "JPEG":
        im.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        im.save(dst, quality=95)
    return orient, im.size


def write_boxes(dst, boxes):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))


def copy_flags(df):
    ph = pd.read_csv(OUT / "phash_matches.csv").set_index(["set", "key"])
    sift_f = OUT / "sift_matches.csv"
    si = pd.read_csv(sift_f).set_index(["set", "key"]) if sift_f.exists() else None
    if si is None:
        print("WARNING: sift_matches.csv missing; only the pHash copy check is applied")
    reason = []
    for s, k in zip(df.set_name, df.src_key):
        r = ""
        if (s, k) in ph.index and ph.at[(s, k), "min_dist_to_train"] <= COPY_DIST:
            r = f"copy of training photo (pHash {ph.at[(s, k), 'min_dist_to_train']})"
        # web photos (MetaDent) get re-posted cropped/edited: eye-checked, 15-24 keypoints were still often the same
        # photo there, so use a lower cut. Clinic sets: 25-27 were different patients, but dropping 2 photos is cheap.
        cut = 15 if s == "metadent" else SIFT_COPY
        if si is not None and (s, k) in si.index and si.at[(s, k), "best_inliers"] >= cut:
            r = f"copy of training photo (SIFT {si.at[(s, k), 'best_inliers']} keypoints)"
        reason.append(r)
    return reason


def groups_from_hash(keys, h, thr=10):
    """Union photos whose pHash distance <= thr (near-duplicates inside a dataset stay in one group)."""
    parent = list(range(len(keys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(keys)):
        d = np.minimum(np.bitwise_count(h[:, 0] ^ h[i, 0]), np.bitwise_count(h[:, 1] ^ h[i, 0]))
        for j in np.nonzero(d <= thr)[0]:
            if j != i:
                parent[find(i)] = find(int(j))
    return {k: find(i) for i, k in enumerate(keys)}


def split_groups(groups, frac_bench, strata=None):
    """Assign whole groups to bench/train: shuffle (seeded), then take frac_bench of each stratum."""
    out = {}
    g = pd.DataFrame({"group": groups, "stratum": strata if strata is not None else 0}).drop_duplicates("group")
    for _, s in g.groupby("stratum"):
        order = rng.permutation(s.group.to_numpy())
        nb = int(round(len(order) * frac_bench))
        out.update({x: ("bench" if i < nb else "train") for i, x in enumerate(order)})
    return [out[x] for x in groups]


# ----------------------------------------------------------------------------------------------------------------
def alphadent():
    A = RAW / "alpha-dent/AlphaDent"
    rows = []
    for split in ["train", "valid"]:
        for p in tqdm(sorted((A / "images" / split).glob("*.jpg")), desc=f"alphadent {split}"):
            pid = p.stem.split("_")[0]
            dst = IMG / "alphadent" / p.name
            with Image.open(p) as im:
                orient, (w, h) = save_img(im, dst)
            boxes = []
            for line in (A / "labels" / split / f"{p.stem}.txt").read_text().splitlines():
                v = line.split()
                if not v or int(v[0]) < 3:     # 0 abrasion, 1 filling, 2 crown: not one of our conditions
                    continue
                pts = np.array(v[1:], float).reshape(-1, 2)
                if orient in ORIENT_PTS:
                    pts = np.array([ORIENT_PTS[orient](x, y) for x, y in pts])
                x0, y0 = pts.min(0).clip(0, 1); x1, y1 = pts.max(0).clip(0, 1)
                boxes.append((1, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))   # 1 = Caries (classes 1-6 merged)
            write_boxes(LAB / "alphadent" / f"{p.stem}.txt", boxes)
            rows.append(dict(id=f"alphadent/{p.stem}", dataset="alphadent", subset=split, group=f"ad_{pid}",
                             image=f"images/alphadent/{p.name}", label=f"labels/alphadent/{p.stem}.txt", width=w, height=h,
                             caries=int(bool(boxes)), n_boxes=len(boxes), exif_orientation=orient,
                             set_name="alphadent", src_key=str(p)))
    df = pd.DataFrame(rows)
    pats = df.drop_duplicates("group")
    # bench = every valid patient + 100 of the 271 train patients (stratified: with / without caries photos)
    tr = pats[pats.subset == "train"].group.to_numpy()
    pick = set(rng.permutation(tr)[:100])
    df["role"] = ["bench" if (s == "valid" or g in pick) else "train" for s, g in zip(df.subset, df.group)]
    return df


def metadent():
    import polars as pl
    md = RAW / "external/MetaDent"
    z = zipfile.ZipFile(md / "labels/metadent_labels.zip")
    cls = json.loads(z.read("metadent/bench/classification.json"))
    meta = pl.concat([pl.read_parquet(f, columns=["filename", "source", "image_type", "contains_eye", "image"])
                      .filter(pl.col("filename").str.replace(".jpg", "", literal=True).is_in(list(cls)))
                      for f in sorted(glob.glob(str(md / "data/*.parquet")))])
    rows = []
    for fn, src, typ, eye, img in tqdm(zip(*[meta[c].to_list() for c in ["filename", "source", "image_type", "contains_eye", "image"]]),
                                       total=meta.height, desc="metadent"):
        k = fn[:-4]
        codes = cls[k]
        dst = IMG / "metadent" / fn
        with Image.open(io.BytesIO(img["bytes"])) as im:
            orient, (w, h) = save_img(im, dst)
        r = dict(id=f"metadent/{k}", dataset="metadent", subset=src, image=f"images/metadent/{fn}", label="",
                 width=w, height=h, codes=" ".join(codes), image_type=typ, face=bool(eye), set_name="metadent", src_key=fn)
        for c, col in MD_MAP.items():
            r[col] = int(c in codes)
        ours = any(c in codes for c in MD_MAP)
        r["problem"] = 1 if ours else (0 if not codes else np.nan)   # out-of-scope-only photos: unknown
        r["out_of_scope_only"] = int(bool(codes) and not ours)
        rows.append(r)
    df = pd.DataFrame(rows)
    h = np.load(OUT / "phash/cand_metadent.npz", allow_pickle=True)
    grp = groups_from_hash(list(h["paths"]), h["h"])
    df["group"] = [f"md_{grp[k]}" for k in df.src_key]
    df["role"] = "train"
    ds3 = df.subset == "DS3"
    strata = df.loc[ds3, "problem"].fillna(2).astype(int).astype(str) + df.loc[ds3, "ulcer"].astype(str)
    df.loc[ds3, "role"] = split_groups(df.loc[ds3, "group"].tolist(), 0.5, strata.tolist())
    df.loc[df.subset == "DS1", "role"] = "bench"
    df.loc[df.ulcer == 1, "role"] = "bench"          # only 18 clean ulcer photos: keep them all for scoring
    # a group can't be split across roles
    df.loc[df.group.isin(df.loc[df.role == "bench", "group"]), "role"] = "bench"
    df["reason"] = ""
    df.loc[df.subset == "DS2", "reason"] = "MetaDent DS2 = the Mendeley set our Kaggle training data came from"
    df.loc[df.image_type != "natural-human", "reason"] = "not a real photo of a person (model / drawing)"
    # face portraits (eyes in view): teeth barely visible, and in DS1 they make up most "healthy" photos, so they
    # would let a model score by "is this a mouth close-up?" instead of by the teeth
    df.loc[df.face & (df.reason == ""), "reason"] = "face portrait, not a mouth photo"
    return df


def autooral():
    A = RAW / "external/Autooral_dataset/Autooral_dataset"
    rows = []
    for split, d, m in [("Train", "data_train", "mask_train"), ("Val", "data_val", "mask_val"), ("Test", "data_test", "mask_test")]:
        xl = next(A.glob(f"{'Training' if split == 'Train' else ('Validation' if split == 'Val' else 'Test')}*.xlsx"))
        info = read_xlsx(xl)
        kinds = ["cancerous", "traumatic", "herpetiform aphthous", "mild aphthous", "severe aphthous"]
        meta = {r[1].zfill(4): (r[0], kinds[r[2:7].index("1")] if "1" in r[2:7] else "") for r in info[1:]}
        for p in tqdm(sorted((A / split / d).glob("*.png")), desc=f"autooral {split}"):
            pid, kind = meta.get(p.stem, ("?", ""))
            dst = IMG / "autooral" / f"{split}_{p.stem}.jpg"
            with Image.open(p) as im:
                _, (w, h) = save_img(im, dst)
            mk = np.asarray(Image.open(A / split / m / p.name).convert("L")) > 127
            n, _, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8))
            H, W = mk.shape
            boxes = [(3, (x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H)          # 3 = Mouth_Ulcer
                     for x, y, bw, bh, a in st[1:] if a >= 0.002 * H * W]
            write_boxes(LAB / "autooral" / f"{split}_{p.stem}.txt", boxes)
            rows.append(dict(id=f"autooral/{split}_{p.stem}", dataset="autooral", subset=split, group=f"ao_{split}_{pid}",
                             image=f"images/autooral/{split}_{p.stem}.jpg", label=f"labels/autooral/{split}_{p.stem}.txt",
                             width=w, height=h, ulcer=1, problem=1, n_boxes=len(boxes), codes=kind,
                             set_name="autooral", src_key=str(p)))
    df = pd.DataFrame(rows)
    df["role"] = "bench"
    tr = df.subset == "Train"
    df.loc[tr, "role"] = split_groups(df.loc[tr, "group"].tolist(), 1 / 3)
    return df


def read_xlsx(p):
    zz = zipfile.ZipFile(p)
    ss = ["".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))
          for si in re.findall(r"<si>(.*?)</si>", zz.read("xl/sharedStrings.xml").decode(), re.S)]
    out = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", zz.read("xl/worksheets/sheet1.xml").decode(), re.S):
        vals = []
        for attrs, inner in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            m = re.search(r"<v>(.*?)</v>", inner or "")
            v = m.group(1) if m else ""
            vals.append(ss[int(v)] if 't="s"' in attrs and v else v)
        out.append(vals)
    return out


def smartom():
    base = RAW / "external/SMART-OM/SMART-OM/01. Normal/01. Unannotated"
    rows = []
    for region in sorted(base.iterdir()):
        if "arch" in region.name.lower():      # teeth photos: teeth are not labelled (many have caries/stains)
            continue
        for p in tqdm(sorted(region.iterdir()), desc=f"smartom {region.name}"):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = re.match(r"(SMITA\d+)", p.name) or re.match(r"(\d+)", p.name)
            pid = m.group(1) if m else p.stem
            name = f"{region.name[:2]}_{p.stem.replace(' ', '')}.jpg"
            dst = IMG / "smartom" / name
            try:
                with Image.open(p) as im:
                    _, (w, h) = save_img(im, dst)
            except Exception:
                continue
            rows.append(dict(id=f"smartom/{name[:-4]}", dataset="smartom", subset=region.name[4:], group=f"om_{pid}",
                             image=f"images/smartom/{name}", label="", width=w, height=h, ulcer=0,
                             set_name="smartom", src_key=str(p)))
    df = pd.DataFrame(rows)
    df["role"] = split_groups(df.group.tolist(), 0.5)
    return df


def main():
    parts = [alphadent(), metadent(), autooral(), smartom()]
    df = pd.concat(parts, ignore_index=True)
    df["reason"] = df.get("reason", pd.Series("", index=df.index)).fillna("")
    cp = copy_flags(df)
    df["reason"] = [r or c for r, c in zip(df.reason, cp)]
    df.loc[df.reason != "", "role"] = "excluded"
    for c in COLS:
        df[c] = df[c].astype("Int64") if c in df else pd.NA
    keep = ["id", "dataset", "subset", "group", "role", "reason", "image", "label", "width", "height", *COLS,
            "out_of_scope_only", "n_boxes", "codes", "exif_orientation", "src_key"]
    df = df[[c for c in keep if c in df]]
    df["src_key"] = df.src_key.str.replace(str(RAW), "data/raw", regex=False).str.replace("\\", "/", regex=False)
    df.to_csv(MANIFEST, index=False)
    print(f"wrote {MANIFEST} ({len(df)} rows)")
    print(df.groupby(["dataset", "role"]).size().unstack(fill_value=0))
    b = df[df.role == "bench"]
    print("\nbench positives per column:")
    print(b.groupby("dataset")[COLS].agg(lambda s: f"{int((s == 1).sum())}/{int(s.notna().sum())}"))
    print("\nexcluded reasons:", df[df.role == "excluded"].reason.str.split(" \\(").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
