"""Cheap photo-quality checks run before the classifier.

The model's robustness tests (results/Daniel/test_metrics.json) show macro-F1 drops on
low-resolution and blurred images, so these checks ask for a retake rather than
silently screening a poor photo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MIN_SIDE = 224
DARK, BRIGHT = 45, 220
BLUR_VAR = 35.0
LOW_CONTRAST = 18.0


def check(im: Image.Image) -> dict:
    w, h = im.size
    g = im.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype=np.float32)

    mean = float(a.mean())
    std = float(a.std())
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    sharp = float(lap.var())

    issues = []
    if min(w, h) < MIN_SIDE:
        issues.append({"code": "small", "severity": "block",
                       "message": f"The photo is very small ({w}×{h}px). Try a photo at least {MIN_SIDE}px on each side."})
    if mean < DARK:
        issues.append({"code": "dark", "severity": "block",
                       "message": "The photo is too dark. Turn on a light or your phone's torch and try again."})
    elif mean > BRIGHT:
        issues.append({"code": "bright", "severity": "warn",
                       "message": "The photo looks overexposed. Avoid pointing the flash straight at your teeth."})
    if sharp < BLUR_VAR:
        issues.append({"code": "blur", "severity": "warn",
                       "message": "The photo looks blurry. Hold the phone steady and tap the screen to focus."})
    if std < LOW_CONTRAST:
        issues.append({"code": "flat", "severity": "warn",
                       "message": "The photo has very little detail. Fill the frame with the area you're worried about."})

    return {
        "ok": not any(i["severity"] == "block" for i in issues),
        "issues": issues,
        "stats": {"width": w, "height": h, "brightness": round(mean, 1), "sharpness": round(sharp, 1)},
    }
