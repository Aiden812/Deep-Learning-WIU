"""Step 1: find benchmark photos that are copies of a photo any team model trained on.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/dedupe_phash.py

Perceptual hash (64-bit DCT pHash) of each photo and of its mirror image, after trimming the grey (114) letterbox
our resize+pad step adds. Compared against every team training folder on this laptop. Hashes are cached in
data/processed/benchmark/phash/*.npz, so a rerun only hashes new folders.

Writes data/processed/benchmark/phash_matches.csv: set, key, min_dist_to_train, closest_train, min_dist_within_set.
Checked by eye on 2026-11-17: distance <= 6 were copies, 8-12 were different photos. build_manifest.py drops <= 8.
This misses crops and rotations; dedupe_sift.py is the second pass for the web photos.
"""
import glob
import io
import json
import os
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RAW, WORKSPACE  # noqa: E402

CACHE = OUT / "phash"
EXTS = (".jpg", ".jpeg", ".png")
REFS = {   # every training folder on this laptop (union of Brandon's, Aiden's and Daniel's training data)
    "ref_pool_v12": WORKSPACE / "datasets/oral_datasetv1.2/pool/images",
    "ref_oral_new": WORKSPACE / "datasets/oral_dataset_new/oral_dataset/images",
    "ref_kaggle_folders": WORKSPACE / "datasets/oral dataset",
    "ref_rf_v1i": WORKSPACE / "datasets/oral-diseases.v1i.yolov11",
}


def trim_pad(a):
    flat = np.abs(a.astype(np.int16) - 114).max(axis=2) <= 3
    rows, cols = ~flat.all(1), ~flat.all(0)
    if rows.sum() < 16 or cols.sum() < 16:
        return a
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return a[r0:r1, c0:c1]


def phash(gray):
    d = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:])).view(">u8")[0]


def hash_img(im):
    im.draft("RGB", (256, 256))
    g = cv2.cvtColor(trim_pad(np.asarray(im.convert("RGB"))), cv2.COLOR_RGB2GRAY)
    return phash(g), phash(g[:, ::-1])


def hash_path(p):
    try:
        with Image.open(p) as im:
            return hash_img(im)
    except Exception:
        return None


def cached(name, make):
    f = CACHE / f"{name}.npz"
    if not f.exists():
        keys, h = make()
        np.savez(f, paths=np.array(keys, dtype=object), h=np.array(h, dtype=np.uint64))
        print(name, len(keys), "hashed", flush=True)
    z = np.load(f, allow_pickle=True)
    return list(z["paths"]), z["h"]


def hash_folder(paths):
    def make():
        with Pool(max(2, os.cpu_count() - 2)) as pool:
            res = pool.map(hash_path, paths, chunksize=64)
        keep = [(str(p), r) for p, r in zip(paths, res) if r is not None]
        return [k for k, _ in keep], [r for _, r in keep]
    return make


def files(d):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)


def metadent_labelled():
    import polars as pl
    lab = set(json.loads(zipfile.ZipFile(RAW / "external/MetaDent/labels/metadent_labels.zip")
                         .read("metadent/bench/classification.json")))
    keys, hs = [], []
    for pq in sorted(glob.glob(str(RAW / "external/MetaDent/data/*.parquet"))):
        df = pl.read_parquet(pq, columns=["filename", "image"])
        df = df.filter(pl.col("filename").str.replace(".jpg", "", literal=True).is_in(list(lab)))
        for fn, img in zip(df["filename"].to_list(), df["image"].to_list()):
            keys.append(fn)
            hs.append(hash_img(Image.open(io.BytesIO(img["bytes"]))))
    return keys, hs


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    refs = {k: cached(k, hash_folder(files(d))) for k, d in REFS.items()}
    A = RAW / "alpha-dent/AlphaDent/images"
    om = RAW / "external/SMART-OM/SMART-OM/01. Normal/01. Unannotated"
    cands = {
        "alphadent": cached("cand_alphadent", hash_folder(files(A / "train") + files(A / "valid"))),
        "autooral": cached("cand_autooral", hash_folder(files(RAW / "external/Autooral_dataset"))),
        "metadent": cached("cand_metadent", metadent_labelled),
        "smartom": cached("cand_smartom", hash_folder(files(om))),
    }
    names = [f"{k}|{p}" for k, (ps, _) in refs.items() for p in ps]
    rh = np.concatenate([h for _, h in refs.values()])
    rows = []
    for cname, (ps, h) in cands.items():
        for i in range(len(ps)):
            d = np.minimum(np.bitwise_count(rh[:, 0] ^ h[i, 0]), np.bitwise_count(rh[:, 1] ^ h[i, 0]))
            j = int(d.argmin())
            w = np.minimum(np.bitwise_count(h[:, 0] ^ h[i, 0]), np.bitwise_count(h[:, 1] ^ h[i, 0]))
            w[i] = 99
            rows.append((cname, ps[i], int(d[j]), names[j], int(w.min())))
    df = pd.DataFrame(rows, columns=["set", "key", "min_dist_to_train", "closest_train", "min_dist_within_set"])
    df = df[~((df.set == "autooral") & df.key.str.contains("mask_"))]
    df.to_csv(OUT / "phash_matches.csv", index=False)
    for s, g in df.groupby("set"):
        print(f"{s}: {len(g)} photos, copies (<= 8): {(g.min_dist_to_train <= 8).sum()}")


if __name__ == "__main__":
    main()
