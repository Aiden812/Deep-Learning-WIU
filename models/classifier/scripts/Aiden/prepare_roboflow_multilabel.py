"""Build the multi-label classifier dataset from the Roboflow "oral-diseases" v2 export, split without leakage.

Usage (from the repo root, inside .venv; needs a CUDA GPU):
    python models/classifier/scripts/Aiden/prepare_roboflow_multilabel.py data/raw/roboflow_oral/oral-diseases.v2i.yolov11.zip data/processed/classifier_multilabel

Why this is more than unzipping:
- Labels. Each photo can show several conditions. A photo's labels are the classes of its boxes plus the folder
  class in its filename (gingivitis is often not boxed, being spread over the gums rather than one spot).
- Copies. The source dataset already holds many cropped / rotated / recoloured copies of each original photo under
  different names, and Roboflow added 3 augmented copies of every training photo on top. Roboflow's own split puts
  copies of the same photo in train and test. Here copies are found in two steps:
    1. DINOv2 (averaged over the 8 flips/rotations) proposes each image's K most similar images;
    2. SIFT keypoints + RANSAC count the keypoints that line up under one affine transform.
  Pairs with >= GROUP_INLIERS (checked by eye: below ~20, groups start chaining different mouths together) and
  Roboflow's own copy groups are joined into groups; one group = one original photo. Copies show the same mouth, so
  every image gets the union of its group's labels.
- Split. Groups are split 70/15/15, balancing each condition. Val and test keep one image per group (preferring a
  copy Roboflow did not augment), so each test image is a distinct original photo. Weaker copies are handled by
  dropping any val/test photo that has >= LEAK_INLIERS matching keypoints with a training image.

Slow steps are cached in <out>/_cache (DINOv2 embeddings, pair inlier counts); delete it to recompute.
Writes <out>/<split folders>/images/*.jpg and <out>/manifest.csv (file, split, group, one 0/1 column per class).
"""
import re, sys, time, zipfile
from pathlib import Path

import cv2
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torchvision import transforms

BOX_CLASSES = ["calculus", "caries", "gingivitis", "hypodontia", "tooth_discolation", "ulcer"]   # data.yaml order
CLASSES = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Tooth_Discoloration", "Mouth_Ulcer"]
K, N_KEYPOINTS = 20, 400
GROUP_INLIERS, LEAK_INLIERS = 40, 10   # chosen by eye on example pairs; at <= 8 split groups chain into one giant group
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_EVAL_GROUP = 20   # split groups with more photos go to train, so val/test fill evenly
SEED = 42
DEV = "cuda"


def read_export(zip_path, out):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        images = [n for n in names if "/images/" in n and not n.endswith("/")]
        if not all((out / n).exists() for n in images):
            z.extractall(out, members=images)
        rows = []
        for f in images:
            photo = re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", f.split("/")[-1])   # Roboflow appends .rf.<hash> to each copy
            prefix = re.match(r"([a-z_]+?)[-_ (]*\d", photo.lower())
            assert prefix and prefix.group(1) in BOX_CLASSES, f"no known class at the start of {f}"
            label_file = re.sub(r"/images/(.*)\.\w+$", r"/labels/\1.txt", f)
            ids = {int(line.split()[0]) for line in z.read(label_file).decode().splitlines() if line.strip()}
            ids.add(BOX_CLASSES.index(prefix.group(1)))
            rows.append(dict(file=f, rf_split=f.split("/")[0], photo=photo, **{c: int(i in ids) for i, c in enumerate(CLASSES)}))
    return pd.DataFrame(rows)


def dino_embeddings(df, root, cache):
    if cache.exists():
        E = np.load(cache)
        assert len(E) == len(df), "cached embeddings do not match the image list: delete _cache"
        return torch.from_numpy(E.astype(np.float32)).to(DEV)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEV).eval().half()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            x = torch.stack([tf(Image.open(root / f).convert("RGB")) for f in df.file[i:i + 64]]).to(DEV).half()
            views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
            e = torch.stack([torch.nn.functional.normalize(model(v).float(), dim=1) for v in views]).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=1).cpu())
            if i % 3200 == 0:
                print(f"  DINOv2 {i}/{len(df)}", flush=True)
    E = torch.cat(out).numpy().astype(np.float16)
    np.save(cache, E)
    return torch.from_numpy(E.astype(np.float32)).to(DEV)


def candidate_pairs(E, photo_id):
    """Each image's K nearest images (DINOv2), as unique pairs i < j from different Roboflow photos."""
    pairs = []
    for i in range(0, len(E), 2048):
        nn_idx = (E[i:i + 2048] @ E.T).topk(K + 1, dim=1).indices[:, 1:].cpu().numpy()
        rows = np.repeat(np.arange(i, i + len(nn_idx)), K)
        pairs.append(np.stack([rows, nn_idx.ravel()], 1))
    p = np.sort(np.concatenate(pairs), 1)
    p = np.unique(p, axis=0)
    return p[photo_id[p[:, 0]] != photo_id[p[:, 1]]]


def sift_features(df, root):
    """RootSIFT descriptors (padded to N_KEYPOINTS) on the GPU, keypoint positions on the CPU."""
    sift = cv2.SIFT_create(nfeatures=N_KEYPOINTS)
    desc = torch.zeros(len(df), N_KEYPOINTS, 128, dtype=torch.float16, device=DEV)
    pts = np.zeros((len(df), N_KEYPOINTS, 2), np.float32)
    count = np.zeros(len(df), int)
    for i, f in enumerate(df.file):
        g = cv2.imread(str(root / f), cv2.IMREAD_GRAYSCALE)
        g = cv2.medianBlur(cv2.resize(g, (384, 384)), 3)   # the blur removes Roboflow's salt-and-pepper noise
        kp, d = sift.detectAndCompute(g, None)
        if d is None:
            continue
        kp, d = kp[:N_KEYPOINTS], d[:N_KEYPOINTS]
        d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))   # RootSIFT: matches better than plain SIFT
        desc[i, :len(d)] = torch.from_numpy(d).to(DEV).half()
        pts[i, :len(kp)] = [k.pt for k in kp]
        count[i] = len(kp)
        if i % 3200 == 0:
            print(f"  SIFT {i}/{len(df)}", flush=True)
    return desc, pts, count


def inlier_counts(pairs, desc, pts, count):
    """Keypoints that line up under one affine transform (RANSAC), per candidate pair."""
    out = np.zeros(len(pairs), int)
    for s in range(0, len(pairs), 1024):
        a, b = pairs[s:s + 1024, 0], pairs[s:s + 1024, 1]
        sim = torch.bmm(desc[a], desc[b].transpose(1, 2)).float()   # unit vectors: dist^2 = 2 - 2 * dot
        valid_b = torch.arange(N_KEYPOINTS, device=DEV)[None] < torch.from_numpy(count[b]).to(DEV)[:, None]
        sim = sim.masked_fill(~valid_b[:, None, :], -2)
        top2, idx = sim.topk(2, dim=2)
        dist = torch.sqrt(torch.clamp(2 - 2 * top2, min=0))
        good = (dist[..., 0] < 0.8 * dist[..., 1]) & (top2[..., 0] > -2)
        good &= torch.arange(N_KEYPOINTS, device=DEV)[None] < torch.from_numpy(count[a]).to(DEV)[:, None]
        good, idx = good.cpu().numpy(), idx[..., 0].cpu().numpy()
        for k in range(len(a)):
            q = np.flatnonzero(good[k])
            if len(q) < 6:
                continue
            _, mask = cv2.estimateAffine2D(pts[a[k], q], pts[b[k], idx[k, q]], method=cv2.RANSAC, ransacReprojThreshold=8)
            out[s + k] = int(mask.sum()) if mask is not None else 0
        if s % 51200 == 0:
            print(f"  RANSAC {s}/{len(pairs)}", flush=True)
    return out


def split_groups(labels, n_photos):
    """Greedy split of groups into train/val/test by photo count, balancing every condition.

    Groups with rare conditions are placed first; each goes to the split with the largest share of its own quota still
    open for the group's conditions. A group larger than MAX_EVAL_GROUP photos always goes to train.
    """
    rng = np.random.default_rng(SEED)
    freq = labels.mul(n_photos, axis=0).sum()
    total = n_photos.sum()
    order = labels.sample(frac=1, random_state=SEED)
    order = order.assign(rarity=(order[CLASSES] / freq).max(axis=1)).sort_values("rarity", ascending=False, kind="stable")
    need = {s: TARGET[s] * freq for s in TARGET}
    left = {s: TARGET[s] * total for s in TARGET}
    split = {}
    for gid, row in order.iterrows():
        w = n_photos[gid]
        conds = [c for c in CLASSES if row[c]]
        options = [s for s in TARGET if s == "train" or (left[s] >= w and w <= MAX_EVAL_GROUP)]
        score = {s: sum(need[s][c] / (TARGET[s] * freq[c]) for c in conds) + left[s] / (TARGET[s] * total)
                 + rng.random() * 1e-6 for s in options}
        s = max(score, key=score.get)
        split[gid] = s
        left[s] -= w
        for c in conds:
            need[s][c] -= w
    return pd.Series(split)


def main(zip_path, out):
    zip_path, out = Path(zip_path), Path(out)
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    t = time.time()
    df = read_export(zip_path, out)
    photo_id = pd.factorize(df.photo)[0]
    print(f"{len(df)} images, {photo_id.max() + 1} Roboflow photos ({time.time() - t:.0f}s)")

    pair_file = cache / f"pairs_k{K}_kp{N_KEYPOINTS}.npz"
    if pair_file.exists():
        z = np.load(pair_file)
        pairs, inl = z["pairs"], z["inliers"]
    else:
        E = dino_embeddings(df, out, cache / "dino.npy")
        pairs = candidate_pairs(E, photo_id)
        del E
        print(f"{len(pairs)} candidate pairs; extracting SIFT ({time.time() - t:.0f}s)")
        desc, pts, count = sift_features(df, out)
        inl = inlier_counts(pairs, desc, pts, count)
        del desc
        np.savez(pair_file, pairs=pairs, inliers=inl)
    print(f"inliers per candidate pair, percentiles 50/90/99: {np.percentile(inl, [50, 90, 99]).astype(int)} ({time.time() - t:.0f}s)")

    # Groups: confirmed copy pairs + Roboflow's own copies (each image linked to a node for its Roboflow photo)
    n, m = len(df), photo_id.max() + 1

    def components(min_inliers):
        e = pairs[inl >= min_inliers]
        rows = np.concatenate([e[:, 0], np.arange(n)])
        cols = np.concatenate([e[:, 1], photo_id + n])
        _, comp = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + m, n + m)), directed=False)
        return pd.factorize(comp[:n])[0]

    df["group"] = components(GROUP_INLIERS)         # one original photo: safe to share labels
    df["split_group"] = components(LEAK_INLIERS)    # also weaker copies: chains across photos, so only used to split
    sizes = df.group.value_counts()
    print(f"{len(sizes)} groups (original photos) | largest {sizes.iloc[0]} images, next {sizes.iloc[1:5].tolist()} | "
          f"groups of 1 photo: {(df.groupby('group').photo.nunique() == 1).sum()}")

    before = df[CLASSES].sum()
    df[CLASSES] = df.groupby("group")[CLASSES].transform("max")
    print("labels added by merging copies:", (df[CLASSES].sum() - before).to_dict())

    sizes = df.groupby("split_group").group.nunique().sort_values(ascending=False)
    print(f"{len(sizes)} split groups (photos linked by >= {LEAK_INLIERS} keypoints) | photos in largest: {sizes.iloc[:5].tolist()}")
    df["split"] = df.split_group.map(split_groups(df.groupby("split_group")[CLASSES].max(), df.groupby("split_group").group.nunique()))
    # val/test: one image per group, preferring one Roboflow did not augment
    df["augmented"] = df.rf_split == "train"
    keep = (df.split == "train") | ~df.sort_values("augmented").duplicated("group").reindex(df.index)
    df = df[keep].drop(columns=["augmented"])

    split_of = df.split.reindex(range(n)).values   # NaN = val/test copy not kept
    kept = pd.notna(split_of)
    is_train = split_of == "train"
    cross = kept[pairs[:, 0]] & kept[pairs[:, 1]] & (is_train[pairs[:, 0]] != is_train[pairs[:, 1]])
    print(f"candidate pairs between train and val/test: with {LEAK_INLIERS // 2}-{LEAK_INLIERS - 1} matching keypoints (possible missed "
          f"copies, check by eye) {(cross & (inl >= LEAK_INLIERS // 2) & (inl < LEAK_INLIERS)).sum()} | "
          f"with >= {LEAK_INLIERS}: {(cross & (inl >= LEAK_INLIERS)).sum()}")

    assert not set(df[df.split == "train"].split_group) & set(df[df.split != "train"].split_group), "a group is in train and val/test"
    assert (df[CLASSES].sum(axis=1) > 0).all(), "an image has no label"
    df[["file", "split", "group", "split_group", *CLASSES]].to_csv(out / "manifest.csv", index=False)
    print(df.groupby("split")[CLASSES].sum().T)
    print(f"images per split: {df.split.value_counts().to_dict()} | groups per split: {df.groupby('split').group.nunique().to_dict()}")
    print(f"done ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
