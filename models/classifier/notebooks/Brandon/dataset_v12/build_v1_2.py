"""Build the oral dataset v1.2 *pool* (every usable photo, processed the same way, no split yet).

What it does, in order:
1. Classes: keep 5, same ids as v1.1 and shared/conditions.yaml (Calculus, Caries, Hypodontia, Mouth_Ulcer,
   Tooth_Discoloration). Gingivitis boxes are deleted. A photo whose ONLY boxes were gingivitis is removed
   (it is not healthy, so it can't be kept as a "no problem" photo).
2. Roboflow swap: all old roboflow_oral rows of v1.1 (stretched 640x640 + augmented copies) are dropped;
   the 9,155 originals from the clean re-export (datasets/oral-diseases.v1i.yolov11) come in instead.
3. Copies: near-duplicate photos are found with a 64-bit difference hash, confirmed by comparing 16x16
   thumbnails. Across sources the copy with the best labels is kept (roboflow > dentalai > zenodo >
   mendeley > external_clean > kaggle_oral); copies inside one source are kept but share a group.
4. Same processing for every source: EXIF dropped, longest side resized to 640, padded (not stretched)
   to 640x640 with grey (114), saved as JPEG quality 90. Box coordinates are moved to match.

Output: datasets/oral_datasetv1.2/pool/{images,labels}/<md5>.{jpg,txt}, pool.csv, removed.csv,
classes.txt, README.md. Splitting into train/val/test is a separate step.

Run:  .venv-train/Scripts/python.exe Deep-Learning-WIU/models/classifier/notebooks/Brandon/dataset_v12/build_v1_2.py
"""

from __future__ import annotations

import hashlib
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "datasets").is_dir())   # folder holding datasets/
DS = ROOT / "datasets"
V11 = DS / "oral_datasetv1.1" / "oral_dataset"
RF = DS / "oral-diseases.v1i.yolov11"
OUT = DS / "oral_datasetv1.2" / "pool"

SIDE = 640
PAD = (114, 114, 114)      # same grey Ultralytics uses for its own letterbox padding
QUALITY = 90

NAMES = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]
# old id -> new id (None = class deleted)
V11_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}                    # unchanged
RF_MAP = {0: 0, 1: 1, 2: None, 3: 2, 4: 4, 5: 3}            # calculus, caries, gingivitis, missing, discol., ulcer
PRIORITY = ["roboflow_oral", "dentalai", "zenodo", "mendeley_noncaries", "external_clean", "kaggle_oral"]

HAM_MAX = 6      # dHash bits that may differ for a candidate copy
THUMB_MAX = 12   # mean |pixel difference| (0-255) of 16x16 grey thumbnails to confirm a copy


def read_boxes(txt: Path, cmap: dict) -> tuple[list, int]:
    """Return (kept rows [cls, cx, cy, w, h], number of rows dropped by the class map)."""
    kept, dropped = [], 0
    if txt.exists():
        for line in txt.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            new = cmap[int(p[0])]
            if new is None:
                dropped += 1
            else:
                kept.append([new, *map(float, p[1:])])
    return kept, dropped


def dhash(g: Image.Image) -> int:
    a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def process(job: tuple[str, str, list]) -> dict:
    """Open one photo, write the 640x640 padded copy and its moved boxes. Runs in a worker process."""
    src, uid, boxes = job
    out_img, out_lbl = OUT / "images" / f"{uid}.jpg", OUT / "labels" / f"{uid}.txt"
    try:
        with Image.open(src) as im:
            orient = im.getexif().get(274, 1)
            if im.format == "JPEG":
                im.draft("RGB", (SIDE * 2, SIDE * 2))   # fast decode; still >= 2x the target size
            im = im.convert("RGB")
            w0, h0 = im.size
    except Exception as e:  # unreadable file
        return {"uid": uid, "error": f"unreadable: {e}"}
    if orient not in (0, 1):
        # A rotation flag means the boxes may have been drawn on the rotated view. Too few to matter (4).
        return {"uid": uid, "error": f"EXIF rotation {orient}"}

    g = im.convert("L")
    thumb = np.asarray(g.resize((16, 16), Image.LANCZOS), dtype=np.uint8).flatten()
    s = SIDE / max(w0, h0)
    nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
    px, py = (SIDE - nw) // 2, (SIDE - nh) // 2
    canvas = Image.new("RGB", (SIDE, SIDE), PAD)
    canvas.paste(im.resize((nw, nh), Image.LANCZOS), (px, py))
    canvas.save(out_img, format="JPEG", quality=QUALITY)   # a new image: no EXIF/metadata carried over

    rows = []
    for c, cx, cy, bw, bh in boxes:
        rows.append(f"{c} {(cx * nw + px) / SIDE:.6f} {(cy * nh + py) / SIDE:.6f} "
                    f"{bw * nw / SIDE:.6f} {bh * nh / SIDE:.6f}")
    out_lbl.write_text("\n".join(rows) + ("\n" if rows else ""))
    return {"uid": uid, "orig_w": w0, "orig_h": h0, "pad_frac": 1 - (nw * nh) / SIDE**2,
            "dhash": dhash(g), "thumb": thumb.tobytes().hex(), "error": ""}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir()

    rows, removed = [], []

    # ---- v1.1 photos from every source except the old roboflow export ----
    v = pd.read_csv(V11 / "labels.csv")
    v = v[v.dataset != "roboflow_oral"]
    for r in v.itertuples():
        lbl = V11 / r.image.replace("images/", "labels/", 1)
        lbl = lbl.with_suffix(".txt")
        boxes, dropped = read_boxes(lbl, V11_MAP)
        base = dict(uid=r.md5, src=str(V11 / r.image), dataset=r.dataset, label_by=r.label_by,
                    boxes_by=r.boxes_by, route=r.route, v11_group=f"v11_{r.group}", v11_split=r.split,
                    rf_split="", source_file=r.source_file, boxes=boxes, had_issue=int(r.has_issue))
        if r.has_issue and not boxes:
            removed.append({**base, "reason": "no boxes left"})
        else:
            rows.append(base)

    # ---- the clean Roboflow re-export ----
    for split in ("train", "valid", "test"):
        for img in sorted((RF / split / "images").iterdir()):
            boxes, dropped = read_boxes(RF / split / "labels" / (img.stem + ".txt"), RF_MAP)
            base = dict(uid=md5(img), src=str(img), dataset="roboflow_oral", label_by="person",
                        boxes_by="human", route="gt", v11_group="", v11_split="", rf_split=split,
                        source_file=f"oral-diseases.v1i.yolov11/{split}/images/{img.name}",
                        boxes=boxes, had_issue=1)
            if not boxes:
                removed.append({**base, "reason": "only gingivitis boxes"})
            else:
                rows.append(base)

    df = pd.DataFrame(rows)
    # identical files (same md5) listed twice: keep the first
    dup = df.uid.duplicated()
    for r in df[dup].to_dict("records"):
        removed.append({**r, "reason": "byte-identical file already in pool"})
    df = df[~dup].reset_index(drop=True)
    print(f"to process: {len(df)}  removed so far: {len(removed)}")

    jobs = list(zip(df.src, df.uid, df.boxes))
    with ProcessPoolExecutor(max_workers=10) as ex:
        res = pd.DataFrame(list(ex.map(process, jobs, chunksize=64)))
    df = df.merge(res, on="uid")
    bad = df.error != ""
    for r in df[bad].to_dict("records"):
        removed.append({**r, "reason": r["error"]})
        for p in (OUT / "images" / f"{r['uid']}.jpg", OUT / "labels" / f"{r['uid']}.txt"):
            p.unlink(missing_ok=True)
    df = df[~bad].reset_index(drop=True)

    # ---- near-duplicate copies ----
    h = np.array([np.uint64(x) for x in df.dhash], dtype=np.uint64)
    th = np.stack([np.frombuffer(bytes.fromhex(t), dtype=np.uint8) for t in df.thumb]).astype(np.int16)
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n_pairs = 0
    for a in range(0, len(h), 2000):
        ham = np.bitwise_count(h[a:a + 2000, None] ^ h[None, :])
        for i, j in zip(*np.nonzero(ham <= HAM_MAX)):
            i += a
            if j <= i:
                continue
            if np.abs(th[i] - th[j]).mean() <= THUMB_MAX:
                n_pairs += 1
                parent[find(i)] = find(j)
    df["dup_cluster"] = [find(i) for i in range(len(df))]
    print("copy pairs found:", n_pairs)

    rank = {d: k for k, d in enumerate(PRIORITY)}
    drop_idx, examples = [], []
    for _, g in df.groupby("dup_cluster"):
        if g.dataset.nunique() < 2:
            continue
        best = min(g.dataset, key=rank.__getitem__)
        losers = g[g.dataset != best]
        drop_idx += list(losers.index)
        examples.append((best, *sorted(losers.dataset.unique())))
    for r in df.loc[drop_idx].to_dict("records"):
        removed.append({**r, "reason": "copy of a photo kept from a better-labelled source"})
        for p in (OUT / "images" / f"{r['uid']}.jpg", OUT / "labels" / f"{r['uid']}.txt"):
            p.unlink(missing_ok=True)
    df = df.drop(index=drop_idx).reset_index(drop=True)
    print("cross-source copies removed:", len(drop_idx), pd.Series(examples).value_counts().head(10).to_dict())

    # ---- groups: photos that must always land in the same split ----
    # union of: v1.1 patient/copy groups, the copy clusters found above, and roboflow's original file name
    key = [f"dup_{c}" for c in df.dup_cluster]
    rf_name = df.source_file.str.extract(r"(?i)/([^/]+?)_(?:jpe?g|png)\.rf\.", expand=False)
    if (df.dataset.eq("roboflow_oral") & rf_name.isna()).any():
        raise ValueError("a Roboflow file name doesn't match the name pattern")
    gp = {k: k for k in set(key)}

    def gfind(x):
        while gp[x] != x:
            gp[x] = gp[gp[x]]
            x = gp[x]
        return x

    for i, r in df.iterrows():
        for other in (r.v11_group, f"rf_{rf_name[i]}" if r.dataset == "roboflow_oral" else ""):
            if other:
                gp.setdefault(other, other)
                gp[gfind(other)] = gfind(key[i])
    df["group"] = pd.Series([gfind(k) for k in key]).astype("category").cat.codes

    # ---- output table ----
    df["image"] = "images/" + df.uid + ".jpg"
    df["label"] = "labels/" + df.uid + ".txt"
    df["has_issue"] = df.boxes.map(lambda b: int(len(b) > 0))
    df["n_boxes"] = df.boxes.map(len)
    for k, n in enumerate(NAMES):
        df[n] = df.boxes.map(lambda b, k=k: int(any(x[0] == k for x in b)))
    df["Normal"] = 1 - df.has_issue
    df["split"] = ""          # decided by the split step
    cols = ["image", "label", "split", "has_issue", "Normal", *NAMES, "n_boxes", "dataset", "label_by",
            "boxes_by", "route", "group", "orig_w", "orig_h", "pad_frac", "v11_split", "rf_split",
            "source_file", "uid"]
    df[cols].sort_values("image").to_csv(OUT / "pool.csv", index=False)
    rem = pd.DataFrame(removed)
    rem[["uid", "dataset", "reason", "source_file"]].to_csv(OUT / "removed.csv", index=False)
    (OUT / "classes.txt").write_text("\n".join(NAMES) + "\n")

    summary = pd.crosstab(df.dataset, df.has_issue.map({0: "healthy", 1: "problem"}), margins=True)
    cls = df.groupby("dataset")[NAMES].sum()
    print(summary, "\n", cls, "\n", rem.reason.value_counts())
    (OUT / "summary.txt").write_text(f"{summary}\n\n{cls}\n\nremoved:\n{rem.groupby(['reason', 'dataset']).size()}\n")


if __name__ == "__main__":
    main()
