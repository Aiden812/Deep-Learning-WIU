"""Where do oral-diseases photos come from, and do test photos have near-copies in train?"""
import pickle
import re
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

RAW = Path(__file__).resolve().parents[5] / "data/raw"
OD = RAW / "oral-diseases.v2i.yolov11"
OUT = Path(__file__).parent / "oral_overlap.pkl"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def h64(im):
    return int(str(imagehash.phash(im)), 16)


def hashes(paths, dihedral=False):
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("L")
        except Exception:
            continue
        if dihedral:  # 8 flips/rotations, to match Roboflow's rotated/flipped train copies
            vs = [im.rotate(a, expand=True) for a in (0, 90, 180, 270)]
            vs += [v.transpose(Image.FLIP_LEFT_RIGHT) for v in vs]
            out.append((p, [h64(v) for v in vs]))
        else:
            out.append((p, h64(im)))
    return out


def files(d):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG and "_mask" not in p.stem)


if __name__ == "__main__":
    res = {
        "od_eval": hashes(files(OD / "valid") + files(OD / "test"), dihedral=True),
        "od_train": hashes(files(OD / "train")),
        "kaggle": hashes(files(RAW / "kaggle_oral")),
        "zenodo": hashes(files(RAW / "zenodo_full" / "Images")),
        "dentalai": hashes(files(RAW / "dentalai")),
    }
    pickle.dump(res, open(OUT, "wb"))
    print({k: len(v) for k, v in res.items()})
