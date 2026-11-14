"""Stage-2 condition detector: localise what the classifier flagged.

When the classifier says a photo shows an issue, the demo runs the YOLO detector
chosen by models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb on the same
image the classifier saw (the intraoral photo or the mouth crop) and returns one box per
finding: Calculus, Caries, Hypodontia, Mouth_Ulcer, Tooth_Discoloration.

The weights file is self-contained; the sidecar best.json written by the notebook next to
it supplies the operating point (`conf`, the confidence that maximised mean F1 on val),
`iou` and `imgsz`. Boxes below `conf` are still returned down to `weak_conf`, flagged
`confident: false`, so the UI can show "weak candidates". For a flagged photo the demo also
asks for its best guesses (`always=3`): if nothing reaches `conf`, the strongest boxes are
returned however weak (down to `floor_conf`, marked `guess: true` below `weak_conf`), so
the photo is always outlined.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ConditionDetector:
    def __init__(self, weights: Path, device: torch.device, weak_conf: float = 0.15):
        os.environ.setdefault("YOLO_VERBOSE", "False")  # before the import: quiet ultralytics
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"detector weights not found at {self.weights}")
        side = self.weights.with_suffix(".json")
        self.meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        self.conf = float(self.meta.get("conf", 0.25))
        self.iou = float(self.meta.get("iou", 0.7))
        self.imgsz = int(self.meta.get("imgsz", 640))
        self.weak_conf = min(weak_conf, self.conf)
        self.floor_conf = 0.001  # lowest score a forced best guess may have (= ultralytics' val conf)
        self.device = device
        self.model = YOLO(str(self.weights), task="detect")
        self.classes = [self.model.names[i] for i in sorted(self.model.names)]
        self.run = self.meta.get("run") or self.weights.stem
        self(Image.new("RGB", (self.imgsz, self.imgsz), (128, 128, 128)))  # warm-up (CUDA init, fuse)

    def __call__(self, im: Image.Image, always: int = 0) -> list[dict]:
        """Boxes on `im` as fractions of its width/height, strongest first.

        Returns everything at/above `weak_conf`. With `always=k`, if nothing reaches the
        operating point `conf`, the k strongest non-overlapping guesses are returned however
        low their confidence (down to `floor_conf`), so a flagged photo still gets a box.
        """
        r = self.model.predict(im, imgsz=self.imgsz, conf=self.floor_conf if always else self.weak_conf,
                               iou=self.iou, device=self.device, verbose=False)[0]
        boxes = []
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            name = self.classes[int(k)]
            boxes.append({"class": name, "display_name": name.replace("_", " "), "conf": round(float(c), 4),
                          "confident": bool(c >= self.conf),
                          "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]})
        boxes.sort(key=lambda d: -d["conf"])
        out = [b for b in boxes if b["conf"] >= self.weak_conf]
        if always and not any(b["confident"] for b in out):
            # Best guesses: NMS is per class, so the same spot can come back once per class —
            # keep only the strongest label per region.
            for b in boxes:
                if len(out) >= always:
                    break
                if b not in out and all(_iou(b["box"], o["box"]) < 0.5 for o in out):
                    out.append(b)
            for b in out:
                b["guess"] = b["conf"] < self.weak_conf
        return out

    def info(self) -> dict:
        return {"enabled": True, "weights": str(self.weights), "run": self.run, "classes": self.classes,
                "conf": self.conf, "weak_conf": self.weak_conf, "floor_conf": self.floor_conf,
                "iou": self.iou, "imgsz": self.imgsz,
                "trained": self.meta.get("trained")}
