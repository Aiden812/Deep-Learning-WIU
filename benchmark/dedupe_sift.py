"""Step 2: second copy check that survives crops, rotations and colour changes (pHash does not).

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/dedupe_sift.py

For every benchmark candidate photo: find its 3 most similar training photos by ImageNet EfficientNet-B0 features
(cosine), add its closest pHash match, then count SIFT keypoints that line up after a RANSAC homography fit (the same
idea as Aiden's find_leaks.py). >= SIFT_COPY inliers = the same photo.

Writes data/processed/benchmark/sift_matches.csv: set, key, best_inliers, best_ref, cos.
Embeddings are cached in data/processed/benchmark/embed/*.npy.
"""
import io
import glob
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RAW, SIFT_COPY  # noqa: E402
from dedupe_phash import trim_pad  # noqa: E402

EMB = OUT / "embed"
SIDE = 512                      # SIFT works on photos shrunk to this longest side


def load_rgb(key, set_name, md_bytes=None):
    if set_name == "metadent":
        im = Image.open(io.BytesIO(md_bytes[key]))
    else:
        im = Image.open(key)
        im.draft("RGB", (SIDE * 2, SIDE * 2))
    a = trim_pad(np.asarray(im.convert("RGB")))
    s = SIDE / max(a.shape[:2])
    return cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1 else a


class DS(torch.utils.data.Dataset):
    def __init__(self, keys, set_name, md_bytes=None):
        self.keys, self.set, self.md = keys, set_name, md_bytes
        self.tf = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Resize((224, 224), antialias=True),
                                         tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        try:
            return self.tf(Image.fromarray(load_rgb(self.keys[i], self.set, self.md)))
        except Exception:
            return torch.zeros(3, 224, 224)


@torch.no_grad()
def embed(name, keys, set_name, model, md_bytes=None):
    f = EMB / f"{name}.npy"
    if f.exists():
        return np.load(f)
    dl = torch.utils.data.DataLoader(DS(keys, set_name, md_bytes), batch_size=64, num_workers=0 if md_bytes else 6)
    out = []
    for x in tqdm(dl, desc=name, leave=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            e = model(x.to(model.device_))
        out.append(torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy().astype(np.float16))
    e = np.concatenate(out)
    np.save(f, e)
    return e


SIFT = cv2.SIFT_create(nfeatures=1500)
BF = cv2.BFMatcher()


def feats(rgb):
    return SIFT.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)


def inliers(fa, fb):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in (p for p in BF.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    ph = pd.read_csv(OUT / "phash_matches.csv")
    refs = {}
    for name in ["ref_pool_v12", "ref_oral_new", "ref_kaggle_folders", "ref_rf_v1i"]:
        refs[name] = list(np.load(OUT / "phash" / f"{name}.npz", allow_pickle=True)["paths"])
    ref_keys = [p for ps in refs.values() for p in ps]

    md_bytes = {}
    import polars as pl
    lab = set(json.loads(zipfile.ZipFile(RAW / "external/MetaDent/labels/metadent_labels.zip").read("metadent/bench/classification.json")))
    for pq in sorted(glob.glob(str(RAW / "external/MetaDent/data/*.parquet"))):
        df = pl.read_parquet(pq, columns=["filename", "image"]).filter(pl.col("filename").str.replace(".jpg", "", literal=True).is_in(list(lab)))
        md_bytes.update({fn: img["bytes"] for fn, img in zip(df["filename"].to_list(), df["image"].to_list())})

    m = tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Identity()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.eval().to(dev); m.device_ = dev

    R = torch.from_numpy(np.concatenate([embed(k, refs[k], "ref", m) for k in refs]).astype(np.float32)).to(dev)
    rows = []
    sift_cache = {}
    for set_name, g in ph.groupby("set"):
        if set_name == "smartom":   # only the soft-tissue folders are used
            g = g[~g.key.str.contains("arch")]
        keys = list(g.key)
        E = torch.from_numpy(embed(f"cand_{set_name}", keys, set_name, m, md_bytes if set_name == "metadent" else None).astype(np.float32)).to(dev)
        sims = E @ R.T
        top = sims.topk(3, dim=1)
        top_i, top_s = top.indices.cpu().numpy(), top.values.cpu().numpy()
        ph_ref = dict(zip(g.key, g.closest_train.str.split("|", n=1).str[1]))
        for i, k in enumerate(tqdm(keys, desc=f"sift {set_name}")):
            fa = feats(load_rgb(k, set_name, md_bytes))
            cands = [(ref_keys[j], float(s)) for j, s in zip(top_i[i], top_s[i])] + [(ph_ref[k], float("nan"))]
            best = (0, "", float("nan"))
            for rk, cs in cands:
                if rk not in sift_cache:
                    sift_cache[rk] = feats(load_rgb(rk, "ref"))
                n = inliers(fa, sift_cache[rk])
                if n > best[0]:
                    best = (n, rk, cs)
            rows.append((set_name, k, *best))
    out = pd.DataFrame(rows, columns=["set", "key", "best_inliers", "best_ref", "cos"])
    out.to_csv(OUT / "sift_matches.csv", index=False)
    for s, g in out.groupby("set"):
        print(f"{s}: {len(g)} photos, same photo (>= {SIFT_COPY} inliers): {(g.best_inliers >= SIFT_COPY).sum()}, "
              f"10-{SIFT_COPY - 1}: {g.best_inliers.between(10, SIFT_COPY - 1).sum()}")


if __name__ == "__main__":
    main()
