"""Helpers for 01_issue_classifier.ipynb.

They live in a .py file (not the notebook) so DataLoader worker processes can import them on
Windows. Jupyter-defined classes can't be pickled into spawned workers.
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def make_cache(src: str, dst: str, side: int) -> bool:
    """Save a copy of `src` whose longest side is at most `side` px. Returns False if unreadable."""
    dst_p = Path(dst)
    if dst_p.exists():
        return True
    try:
        with Image.open(src) as im:
            if im.format == "JPEG":
                im.draft("RGB", (side * 2, side * 2))  # fast JPEG downscale while decoding
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((side, side), Image.LANCZOS)
            tmp = dst_p.with_suffix(".tmp")
            im.save(tmp, format="JPEG", quality=95)
            tmp.replace(dst_p)  # write then rename, so an interrupted run never leaves half a file
        return True
    except Exception:
        return False


def make_cache_nopad(src: str, dst: str, side: int, orig_w: int, orig_h: int, canvas: int = 640) -> bool:
    """Like make_cache, but for a v1.2 pool photo: first cut off the grey padding.

    The pool photos are the original resized to fit `canvas` and centred on grey. The photo's original size
    tells exactly where it sits, so the padding is removed without guessing. The model then never sees the
    padding, and so can't use how much padding there is as a clue.
    """
    dst_p = Path(dst)
    if dst_p.exists():
        return True
    try:
        s = canvas / max(orig_w, orig_h)
        nw, nh = max(1, round(orig_w * s)), max(1, round(orig_h * s))
        px, py = (canvas - nw) // 2, (canvas - nh) // 2
        with Image.open(src) as im:
            im = im.convert("RGB").crop((px, py, px + nw, py + nh))
            im.thumbnail((side, side), Image.LANCZOS)
            tmp = dst_p.with_suffix(".tmp")
            im.save(tmp, format="JPEG", quality=95)
            tmp.replace(dst_p)
        return True
    except Exception:
        return False


# ---- photo corruptions for the robustness test ------------------------------------------

def blur(im: Image.Image) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(2))


def dark(im: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(0.5)


def bright(im: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(1.6)


def low_res(im: Image.Image) -> Image.Image:
    w, h = im.size
    return im.resize((max(1, w // 3), max(1, h // 3)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def jpeg20(im: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=20)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {"blur": blur, "dark (x0.5)": dark, "bright (x1.6)": bright,
               "low-res (1/3)": low_res, "JPEG q=20": jpeg20}


class OralDS(torch.utils.data.Dataset):
    """(image tensor, has_issue) pairs from cached JPEGs. `corrupt` names an entry in CORRUPTIONS."""

    def __init__(self, paths: list[str], labels: list[int], tf, corrupt: str | None = None):
        self.paths, self.labels, self.tf, self.corrupt = list(paths), list(labels), tf, corrupt

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        if self.corrupt:
            im = CORRUPTIONS[self.corrupt](im)
        return self.tf(im), self.labels[i]
