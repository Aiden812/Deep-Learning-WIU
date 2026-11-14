"""Find photos in the v1.2 pool that are copies of a Roboflow photo even if mirrored or recoloured.

Each photo -> padding cropped off -> 32x32 grey -> normalised (removes brightness/contrast changes).
Score = best correlation with any Roboflow photo, as-is or mirrored left-right. 1.0 = identical.
Writes pool/rf_copy_scores.csv and a contact sheet per score band to choose the cut-off by eye.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

P = next(p for p in Path(__file__).resolve().parents if (p / "datasets").is_dir()) / "datasets" / "oral_datasetv1.2" / "pool"


def vec(path: str) -> np.ndarray:
    im = Image.open(P / path).convert("L")
    a = np.asarray(im).astype(np.int16)
    ys, xs = np.where(np.abs(a - 114) > 3)
    if len(xs):
        im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    v = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32)
    return v


def norm(v: np.ndarray) -> np.ndarray:
    v = v.reshape(len(v), -1)
    v = v - v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)


def main() -> None:
    d = pd.read_csv(P / "pool.csv")
    with ProcessPoolExecutor(10) as ex:
        V = np.stack(list(ex.map(vec, d.image, chunksize=64)))
    rf = (d.dataset == "roboflow_oral").to_numpy()
    R = norm(V[rf])
    Rf = norm(V[rf][:, :, ::-1].copy())
    O = norm(V[~rf])
    s1, s2 = O @ R.T, O @ Rf.T
    best = np.maximum(s1, s2)
    j = best.argmax(1)
    other = d[~rf].copy()
    other["rf_score"] = best[np.arange(len(j)), j]
    other["rf_match"] = d[rf].image.to_numpy()[j]
    other["mirrored"] = s2[np.arange(len(j)), j] > s1[np.arange(len(j)), j]
    other[["image", "dataset", "has_issue", "rf_score", "rf_match", "mirrored"]].sort_values(
        "rf_score", ascending=False).to_csv(P / "rf_copy_scores.csv", index=False)
    print(other.groupby("dataset").rf_score.describe(percentiles=[.5, .9, .99]).round(3))
    for lo in (0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80):
        print(lo, other[other.rf_score >= lo].dataset.value_counts().to_dict())

    # contact sheets: 8 pairs from each band
    bands = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85)]
    sheet = Image.new("RGB", (len(bands) * 340, 8 * 165), "white")
    for b, (lo, hi) in enumerate(bands):
        g = other[(other.rf_score >= lo) & (other.rf_score < hi)]
        g = g.sample(min(8, len(g)), random_state=0)
        for k, r in enumerate(g.itertuples()):
            sheet.paste(Image.open(P / r.image).resize((160, 160)), (b * 340, k * 165))
            sheet.paste(Image.open(P / r.rf_match).resize((160, 160)), (b * 340 + 165, k * 165))
    sheet.save(P.parent / "rf_copy_bands.png")


def apply_review() -> None:
    """Remove the copies confirmed by eye (listed in datasets/oral_datasetv1.2/rf_copies_reviewed.csv)."""
    reviewed = pd.read_csv(P.parent / "rf_copies_reviewed.csv")
    p, r = pd.read_csv(P / "pool.csv"), pd.read_csv(P / "removed.csv")
    rows = p[p.image.isin(reviewed.image)]
    r = pd.concat([r, rows.assign(reason="copy of a Roboflow photo (mirrored/recoloured/rotated), checked by eye")[r.columns]])
    for im, lb in zip(rows.image, rows.label):
        (P / im).unlink()
        (P / lb).unlink()
    p[~p.image.isin(reviewed.image)].to_csv(P / "pool.csv", index=False)
    r.to_csv(P / "removed.csv", index=False)
    print(f"removed {len(rows)} of {len(reviewed)} reviewed copies")


if __name__ == "__main__":
    import sys
    apply_review() if "--apply" in sys.argv else main()
