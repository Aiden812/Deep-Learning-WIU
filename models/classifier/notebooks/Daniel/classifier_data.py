"""Dataset and photo degradations for oral_condition_classifier.ipynb.

They live in a .py file, not in the notebook, so DataLoader worker processes can load them: on Windows each worker is a
fresh Python process that imports what it needs, and it cannot import classes or lambdas defined in a notebook.
"""
import io

from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


class ImageDS(Dataset):
    """Images from `paths` with integer `labels`, passed through `tf`.

    Plain torch Dataset used by both the notebook and this module's helpers. It deliberately holds only
    paths and plain functions (never lambdas), so DataLoader workers on Windows can pickle it.

    `degrade` is None (no degradation), one function applied to every image (train-time augmentation of
    the whole loader), or a list with one function (or None) per image - used to build the validation
    copy where each image gets a different, fixed degradation so the degraded scores stay comparable
    across epochs and runs.
    """

    def __init__(self, paths, labels, tf, degrade=None):
        """Store the sample list and transforms; nothing is loaded here (paths only, lazily read per item)."""
        self.paths, self.labels, self.tf, self.degrade = list(paths), list(labels), tf, degrade

    def __len__(self):
        """Number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, i):
        """Image i as (transformed tensor, label).

        Read order matters: open + RGB-convert, then degrade (simulating a bad photo BEFORE the resize
        and augmentation of `tf`, which is what the model would really see), then `tf`.
        """
        im = Image.open(self.paths[i]).convert("RGB")
        # per-image degradation list (validation copy) or one function for the whole batch
        fn = self.degrade[i] if isinstance(self.degrade, (list, tuple)) else self.degrade
        if fn is not None:
            im = fn(im)
        return self.tf(im), self.labels[i]


def jpeg_roundtrip(im, q):
    """Re-save `im` through an in-memory JPEG at quality `q` and return the re-read image.

    Going through a real encoder/decoder (instead of a pixel filter) adds the actual blocking and
    ringing artefacts of JPEG, which is what photos picked up from chat apps or web searches carry.
    `convert("RGB")` after the read drops the alpha channel JPEGs cannot store anyway and gives the
    same mode back as the input.
    """
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def lowres(im):
    """Web-thumbnail quality: shrink the longest side to 160 px, then JPEG at quality 60.

    This models the most common real failure for scraped images: a site's small preview/ thumbnail
    downloaded instead of the full photo. Bilinear downscale approximates a browser resample.
    """
    w, h = im.size
    s = 160 / max(w, h)
    return jpeg_roundtrip(im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR), 60)


def dark(im):
    """Half brightness: a poorly lit phone photo (dim room, no flash).

    Applied as a brightness multiplier so relative contrast is preserved - the image goes dark the way
    an underexposed photo does, not like it was covered with grey paint.
    """
    return ImageEnhance.Brightness(im).enhance(0.5)


def bright(im):
    """1.6x brightness: over-exposed / flash-washed photo, where highlights clip toward white."""
    return ImageEnhance.Brightness(im).enhance(1.6)


def blur(im):
    """Gaussian blur with radius 2 px: an out-of-focus or hand-shaken shot.

    Radius is relative to the original photo size; after the loader's resize to 224 px it stays mild,
    roughly the level where teeth are still recognisable but fine texture is gone.
    """
    return im.filter(ImageFilter.GaussianBlur(2))


def jpeg20(im):
    """Heavy JPEG compression (quality 20): the aggressive re-compression chat apps apply when forwarding photos."""
    return jpeg_roundtrip(im, 20)


# Everything the notebook's "phone quality" checks iterate over: display name -> degradation (None = untouched).
DEGRADATIONS = {
    "original": None,
    "low-res 160px + JPEG60": lowres,
    "dark (x0.5)": dark,
    "bright (x1.6)": bright,
    "blur (r=2)": blur,
    "JPEG q=20": jpeg20,
}
# "Phone-quality" validation copy: each val image gets ONE degradation, cycling evenly through these four:
# lowres (resolution), dark (exposure), blur (sharpness) and jpeg20 (compression) - four independent failure
# modes. `bright` is tested as a standalone option in DEGRADATIONS but is not part of the val cycle.
VAL_DEGRADE_CYCLE = [lowres, dark, blur, jpeg20]


def val_degrade_list(n):
    """One degradation per image for n validation images, cycling so all four types appear evenly.

    The cycle is positional (image i -> cycle[i % 4]), so the same validation list always produces the
    same degraded copy - two runs of the robustness check are directly comparable.
    """
    return [VAL_DEGRADE_CYCLE[i % len(VAL_DEGRADE_CYCLE)] for i in range(n)]
