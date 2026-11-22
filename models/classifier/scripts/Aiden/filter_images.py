"""Automatic quality checks for the oral image dataset.

Usage:  python filter_images.py <dataset_folder> <output_folder>

Expects one sub-folder per class (any depth; the image's parent folder name is
its class). It never moves or deletes images. It writes:
  report.csv          one row per image with its measurements and flags
  sheets/<class>_NN.jpg  numbered thumbnail grids for the human review step
Images with no flags still need the human review (e.g. tongue hiding the teeth),
which no simple measurement can catch.
"""
import csv, sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIDE = 128        # smaller than this is too little detail to learn from
DARK, BRIGHT = 40, 220  # mean grey level outside this range = badly exposed
BLUR_MIN = 30.0       # variance of Laplacian; lower = blurry. Check sheets before trusting.
GREY_MAX = 15.0       # mean HSV saturation; lower = greyscale photo or X-ray, not a phone photo
DUP_BITS = 4          # perceptual hashes this close = same photo (resized, re-saved or flipped copy)
TILE, COLS, ROWS = 160, 8, 6


def measure(path):
    """Return the measurements for one image, or None if it cannot be opened."""
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return None
    rgb = np.asarray(img)
    small = cv2.resize(rgb, (512, 512 * rgb.shape[0] // rgb.shape[1] or 1))
    grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    return {
        "width": img.width, "height": img.height,
        "brightness": round(float(grey.mean()), 1),
        "blur": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 1),
        "saturation": round(float(hsv[..., 1].mean()), 1),
        "hash": imagehash.phash(img),
        "hash_flip": imagehash.phash(img.transpose(Image.FLIP_LEFT_RIGHT)),
    }


def duplicate_groups(hashes, flips):
    """Group images whose hashes (or mirrored hashes) differ by <= DUP_BITS bits."""
    bits = np.array([h.hash.flatten() for h in hashes], dtype=np.int32)
    fbits = np.array([h.hash.flatten() for h in flips], dtype=np.int32)
    def hamming(a, b):  # pairwise bit differences, done as matrix products so 5k images is quick
        return a @ (1 - b).T + (1 - a) @ b.T
    close = (hamming(bits, bits) <= DUP_BITS) | (hamming(bits, fbits) <= DUP_BITS)
    parent = list(range(len(hashes)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(close, 1))):
        parent[root(j)] = root(i)
    return [root(i) for i in range(len(hashes))]


def contact_sheets(rows, out):
    """Numbered thumbnail grids per class so a person can review every image quickly."""
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    per = COLS * ROWS
    for cls, items in by_class.items():
        for s in range(0, len(items), per):
            sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + 18)), "white")
            draw = ImageDraw.Draw(sheet)
            for k, r in enumerate(items[s:s + per]):
                x, y = (k % COLS) * TILE, (k // COLS) * (TILE + 18)
                r["sheet"] = f"{cls}_{s // per + 1:02d}"
                r["tile"] = k + 1
                try:
                    im = Image.open(r["path"]).convert("RGB")
                    im.thumbnail((TILE - 4, TILE - 4))
                    sheet.paste(im, (x + 2, y + 2))
                except Exception:
                    pass
                draw.text((x + 4, y + TILE + 2), f"{k + 1} {r['flags'][:18]}", fill="red" if r["flags"] else "black")
            sheet.save(out / "sheets" / f"{r['sheet']}.jpg", quality=80)


def main(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ok = [], []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        m = measure(p)
        row = {"path": str(p), "class": p.parent.name, "flags": ""}
        if m is None:
            row["flags"] = "corrupt"
        else:
            flags = []
            if min(m["width"], m["height"]) < MIN_SIDE: flags.append("too_small")
            if m["brightness"] < DARK: flags.append("too_dark")
            if m["brightness"] > BRIGHT: flags.append("too_bright")
            if m["blur"] < BLUR_MIN: flags.append("blurry")
            if m["saturation"] < GREY_MAX: flags.append("greyscale_or_xray")
            row.update(m, flags=";".join(flags))
            ok.append(row)
        rows.append(row)
    if not rows:
        sys.exit(f"No images found under {src}")

    groups = duplicate_groups([r["hash"] for r in ok], [r["hash_flip"] for r in ok]) if ok else []
    members = {}
    for r, g in zip(ok, groups):
        members.setdefault(g, []).append(r)
    for g, rs in members.items():
        if len(rs) < 2:
            continue
        keep = rs[0]
        mixed = len({r["class"] for r in rs}) > 1
        for r in rs:
            r["dup_group"] = g
            extra = "label_conflict" if mixed else ("" if r is keep else "duplicate")
            r["flags"] = ";".join(f for f in (r["flags"], extra) if f)

    contact_sheets(rows, out)
    cols = ["path", "class", "flags", "dup_group", "width", "height", "brightness", "blur", "saturation", "sheet", "tile"]
    with open(out / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} images, {sum(1 for r in rows if r['flags'])} flagged")
    counts = {}
    for r in rows:
        for fl in filter(None, r["flags"].split(";")):
            counts[fl] = counts.get(fl, 0) + 1
    for fl, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {fl:20s} {n}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
