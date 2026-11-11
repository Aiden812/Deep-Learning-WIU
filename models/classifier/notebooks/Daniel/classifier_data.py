"""Dataset and photo degradations for oral_condition_classifier.ipynb.

They live in a .py file, not in the notebook, so DataLoader worker processes can load them: on Windows each worker is a
fresh Python process that imports what it needs, and it cannot import classes or lambdas defined in a notebook.
"""
import io

from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


class ImageDS(Dataset):
    """Images from `paths` with integer `labels`, passed through `tf`.

    `degrade` is None, one function for every image, or a list with one function (or None) per image.
    """

    def __init__(self, paths, labels, tf, degrade=None):
        self.paths, self.labels, self.tf, self.degrade = list(paths), list(labels), tf, degrade

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        fn = self.degrade[i] if isinstance(self.degrade, (list, tuple)) else self.degrade
        if fn is not None:
            im = fn(im)
        return self.tf(im), self.labels[i]


def jpeg_roundtrip(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def lowres(im):
    # web-thumbnail quality: longest side 160 px, JPEG quality 60
    w, h = im.size
    s = 160 / max(w, h)
    return jpeg_roundtrip(im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR), 60)


def dark(im):
    return ImageEnhance.Brightness(im).enhance(0.5)


def bright(im):
    return ImageEnhance.Brightness(im).enhance(1.6)


def blur(im):
    return im.filter(ImageFilter.GaussianBlur(2))


def jpeg20(im):
    return jpeg_roundtrip(im, 20)


DEGRADATIONS = {
    "original": None,
    "low-res 160px + JPEG60": lowres,
    "dark (x0.5)": dark,
    "bright (x1.6)": bright,
    "blur (r=2)": blur,
    "JPEG q=20": jpeg20,
}
# "Phone-quality" validation copy: each val image gets one degradation, cycling through these four
VAL_DEGRADE_CYCLE = [lowres, dark, blur, jpeg20]


def val_degrade_list(n):
    return [VAL_DEGRADE_CYCLE[i % len(VAL_DEGRADE_CYCLE)] for i in range(n)]
