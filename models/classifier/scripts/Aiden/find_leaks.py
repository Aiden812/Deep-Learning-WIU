"""Mark which val/test images of Daniel's oral_dataset are copies of a training photo.

    python models/classifier/scripts/Aiden/find_leaks.py

Daniel's split keeps copies together by Roboflow name and perceptual hash, but Roboflow's own copies are often cropped,
rotated or recoloured, which those checks miss. This script reuses the copy matching from prepare_roboflow_multilabel.py
(DINOv2 candidates confirmed by SIFT + RANSAC keypoints, cached in data/processed/classifier_multilabel/_cache), so it
needs no GPU. A val/test image counts as leaked when it is in the same copy group as a training image (>= 40 matching
keypoints, or the same Roboflow photo name), or matches a training image directly with >= 20 keypoints (different photos
reach up to ~20; see decisions.md, 2026-11-05).

Only roboflow_oral images can be checked this way. In val/test those are every image with a condition; the healthy
(Normal) images come from other sources and are kept as they are.

Writes data/processed/oral_dataset_leaks.csv: md5, split, checked, leaked, max_keypoints_to_train.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[4]
DATASET = REPO / "data/raw/oral_dataset"
CACHE = REPO / "data/processed/classifier_multilabel/_cache/pairs_k20_kp400.npz"
ZIP_CANDIDATES = [REPO / "data/raw/roboflow_oral/oral-diseases.v2i.yolov11.zip", REPO.parent / "oral-diseases.v2i.yolov11.zip"]
OUT = REPO / "data/processed/oral_dataset_leaks.csv"
GROUP_KP, DIRECT_KP = 40, 20   # keypoint cut-offs: >= 40 = surely the same photo; >= 20 = copy (different photos stay below)


def main():
    # Matching is precomputed: DINOv2 found look-alike pairs, then SIFT counted keypoints that line up between them
    # after a geometric fit (RANSAC). That survives crops, rotations and colour changes, unlike a perceptual hash.
    zip_path = next((p for p in ZIP_CANDIDATES if p.exists()), None)
    assert zip_path, f"Roboflow zip not found in {ZIP_CANDIDATES}"
    assert CACHE.exists(), f"{CACHE} missing: run prepare_roboflow_multilabel.py once first"
    # same image order as prepare_roboflow_multilabel.read_export, which the cached pair indices refer to
    images = [n for n in zipfile.ZipFile(zip_path).namelist() if "/images/" in n and not n.endswith("/")]
    key = pd.Series([n.split("/")[-1] for n in images])
    # Roboflow names augmented copies "<photo>.rf.<hash>.jpg": strip the hash so copies of one photo share an id
    photo = pd.factorize(key.str.replace(r"\.rf\.[0-9a-f]+\.\w+$", "", regex=True))[0]
    c = np.load(CACHE)
    pairs, kp = c["pairs"], c["inliers"]
    assert pairs.max() < len(images), "cache does not match this zip"

    # which split (train / val / test) Daniel put each Roboflow image in
    d = pd.read_csv(DATASET / "labels.csv", keep_default_na=False)
    rf = d[d.dataset == "roboflow_oral"].assign(key=lambda x: x.source_file.str.split("/").str[-1])
    assert rf.key.is_unique
    split = rf.set_index("key").split.reindex(key).to_numpy()   # NaN = image Daniel left out
    is_train = split == "train"

    # Build copy groups: a graph where images are linked if they share >= 40 keypoints or the same photo name.
    # Connected components = groups of copies. A val/test image whose group contains a training image is leaked.
    n, m = len(images), photo.max() + 1
    strong = pairs[kp >= GROUP_KP]
    rows = np.concatenate([strong[:, 0], np.arange(n)])
    cols = np.concatenate([strong[:, 1], photo + n])              # same Roboflow photo name = same photo
    group = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + m, n + m)), directed=False)[1][:n]
    leaked = np.isin(group, np.unique(group[is_train]))
    best = np.zeros(n, int)                                        # most keypoints matched with any training image
    for a, b in ((0, 1), (1, 0)):
        to_train = is_train[pairs[:, b]]
        np.maximum.at(best, pairs[to_train, a], kp[to_train])
    leaked |= best >= DIRECT_KP   # also leaked: a direct match of >= 20 keypoints with any training image

    # Map the result back to every image of oral_dataset (non-Roboflow images can't be checked: leaked = False)
    info = pd.DataFrame({"key": key, "leaked": leaked, "max_keypoints_to_train": best})
    out = d[["md5", "split", "dataset", "source_file"]].copy()
    out["key"] = np.where(out.dataset == "roboflow_oral", out.source_file.str.split("/").str[-1], None)
    out = out.merge(info, on="key", how="left")
    out["checked"] = out.dataset == "roboflow_oral"
    out["leaked"] = out.leaked.fillna(False).astype(bool) & (out.split != "train")
    out["max_keypoints_to_train"] = out.max_keypoints_to_train.fillna(0).astype(int)
    out[["md5", "split", "checked", "leaked", "max_keypoints_to_train"]].to_csv(OUT, index=False)

    ev = out[out.split != "train"]
    print(ev.groupby(["split", "checked"]).leaked.agg(images="size", leaked="sum").to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
