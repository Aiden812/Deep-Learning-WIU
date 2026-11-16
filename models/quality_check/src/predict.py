"""Photo quality gate: runs before the classifier / detector, CPU only (PIL + numpy, no model).

    check(image) -> {"ok": bool, "status": "pass" | "warn" | "fail", "reasons": [...], "checks": [...]}

Each check is {"id", "label", "status", "value", "tip"}. "fail" means the classifier must not guess on the
photo; "warn" means it still runs, but the patient is told the result may be less reliable.

Thresholds were set on sample photos from the classifier dataset plus darkened / blurred / washed-out copies
(see models/quality_check/README.md). They are simple measurements, so they can be wrong on unusual photos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

WORK_SIZE = 512  # measurements are taken on a copy whose long side is this, so big and small photos compare fairly

# (id, label shown in the checklist, tip shown when it fails or warns)
CHECKS = [
    ("size", "Big enough", "The photo is very small. Use your camera's normal photo mode, not a thumbnail or screenshot."),
    ("light", "Bright enough", "It's too dark. Try near a window or turn on a light."),
    ("exposure", "Not washed out", "It's too bright or washed out. Don't point the camera at a lamp."),
    ("glare", "No flash glare", "Flash glare is hiding part of your teeth. Try without flash, or tilt the phone a little."),
    ("focus", "In focus", "It's blurry. Hold the phone still and tap your teeth on screen to focus."),
    ("contrast", "Clear detail", "We can't make out any detail. Get closer and make sure the lens is clean."),
    ("mouth", "Shows teeth and gums", "We can't see teeth or gums. Pull your lips back and fill the frame with your mouth."),
]
LABELS = {i: (label, tip) for i, label, tip in CHECKS}

# status thresholds: (fail_if, warn_if)
MIN_SIDE = (100, 160)          # px, short side of the original (training photos are small web images, median ~285)
LIT_LOW = (60, 95)             # 0-255, 90th-percentile brightness: are the brightest parts (the teeth) lit?
                               # Not the average, so a dark background around a lit mouth is fine.
MEAN_LUM_HIGH = (225, 205)     # 0-255, whole picture too bright
CLIPPED = (1.01, 0.15)         # fraction of pure-white pixels (all channels >= 250); warns only, never blocks
SHARPNESS = (10.0, 35.0)       # second-sharpest-block Laplacian variance on the 512 px copy
CONTRAST = (14.0, 22.0)        # std of luminance
MOUTH = (0.15, 0.35)           # fraction of pixels coloured like gums / lips / tongue / teeth


def _work_copy(image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = image.size
    scale = WORK_SIZE / max(size)
    if scale < 1:
        image = image.resize((max(1, round(size[0] * scale)), max(1, round(size[1] * scale))), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32), size


def _sharpness(lum: np.ndarray) -> float:
    """Variance of the Laplacian in the sharpest block of a 4x4 grid. Teeth often fill only part of the frame,
    and a soft background should not make a sharp close-up count as blurry."""
    lap = (lum[1:-1, :-2] + lum[1:-1, 2:] + lum[:-2, 1:-1] + lum[2:, 1:-1] - 4 * lum[1:-1, 1:-1])
    h, w = lap.shape
    if h < 8 or w < 8:
        return float(lap.var())
    blocks = [lap[r * h // 4:(r + 1) * h // 4, c * w // 4:(c + 1) * w // 4].var() for r in range(4) for c in range(4)]
    # second-sharpest block, so one patch of sensor noise or text can't pass a blurry photo on its own
    return float(sorted(blocks)[-2])


def _mouth_fraction(rgb: np.ndarray) -> float:
    """Share of pixels that look like gum / lip / tongue (pink-red) or tooth (pale, warm, low saturation)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    pinkred = (r >= g) & (r >= b) & (r - g > 25) & (sat > 0.18) & (mx > 60) & (b < r * 0.95)
    tooth = (mx > 140) & (sat < 0.35) & (r >= b) & (mn > 90)
    return float((pinkred | tooth).mean())


def _grade(value: float, limits: tuple[float, float], higher_is_better: bool = True) -> str:
    fail, warn = limits
    if higher_is_better:
        return "fail" if value < fail else "warn" if value < warn else "pass"
    return "fail" if value > fail else "warn" if value > warn else "pass"


def check(image: Image.Image) -> dict:
    rgb, (w, h) = _work_copy(image)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mean = float(lum.mean())
    lit = float(np.percentile(lum, 90))
    clipped = float((rgb.min(axis=-1) >= 250).mean())

    values = {
        "size": min(w, h),
        "light": round(lit, 1),
        "exposure": round(mean, 1),
        "glare": round(clipped, 3),
        "focus": round(_sharpness(lum), 1),
        "contrast": round(float(lum.std()), 1),
        "mouth": round(_mouth_fraction(rgb), 3),
    }
    status = {
        "size": _grade(values["size"], MIN_SIDE),
        "light": _grade(lit, LIT_LOW),
        "exposure": _grade(mean, MEAN_LUM_HIGH, higher_is_better=False),
        "glare": _grade(clipped, CLIPPED, higher_is_better=False),
        "focus": _grade(values["focus"], SHARPNESS),
        "contrast": _grade(values["contrast"], CONTRAST),
        "mouth": _grade(values["mouth"], MOUTH),
    }
    # In a dim photo the colours fade too, so "no teeth" is really "too dark": blame the light.
    if status["light"] == "warn" and status["mouth"] == "fail":
        status["light"] = "fail"
    # A very dark or washed-out photo also looks flat, blurry and colourless. Report only the cause.
    if status["light"] == "fail" or status["exposure"] == "fail":
        for k in ("focus", "contrast", "mouth", "glare"):
            if status[k] != "pass":
                status[k] = "skip"
    elif status["contrast"] == "fail" and status["focus"] != "pass":
        status["focus"] = "skip"  # a featureless photo is "no detail", not "hold still"

    checks = [{"id": i, "label": LABELS[i][0], "status": status[i], "value": values[i],
               "tip": LABELS[i][1] if status[i] in ("fail", "warn") else None} for i, *_ in CHECKS]
    fails = [c["id"] for c in checks if c["status"] == "fail"]
    warns = [c["id"] for c in checks if c["status"] == "warn"]
    return {
        "ok": not fails,
        "status": "fail" if fails else "warn" if warns else "pass",
        "reasons": fails + warns,
        "checks": checks,
    }


def predict(image_path: Path, model=None) -> dict:
    with Image.open(image_path) as im:
        return check(im)


if __name__ == "__main__":
    import json
    import sys

    for p in sys.argv[1:]:
        r = predict(Path(p))
        print(p, r["status"], json.dumps({c["id"]: [c["status"], c["value"]] for c in r["checks"]}))
