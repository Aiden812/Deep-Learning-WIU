"""Any ultralytics YOLO detector. Photo-level score = highest box confidence (overall, and per condition);
flag = at or above the model's own operating conf (a best.json sidecar's "conf" if present, else 0.25 as in
Brandon's notebook). Boxes are kept down to conf 0.001 so AP can be computed."""
import json

import numpy as np

from common import CLASSES
from . import to_ours


class Yolo:
    batchable = True

    def __init__(self, weights, conf=None):
        from ultralytics import YOLO
        side = weights.with_suffix(".json")
        meta = json.loads(side.read_text()) if side.exists() else {}
        self.conf = float(meta.get("conf", conf if conf is not None else 0.25))
        self.model = YOLO(str(weights), task="detect")
        self.imgsz = int(meta.get("imgsz", (self.model.ckpt or {}).get("train_args", {}).get("imgsz", 640)))
        self.map = {i: (CLASSES.index(to_ours(n)) if to_ours(n) else None) for i, n in self.model.names.items()}

    def predict_batch(self, ims):
        out = []
        import torch
        dev = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        for r in self.model.predict(ims, imgsz=self.imgsz, conf=0.001, max_det=100, verbose=False, device=dev):
            boxes = [[self.map[int(k)], float(c), *map(float, b)]
                     for b, c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                     if self.map[int(k)] is not None]
            top = max((b[1] for b in boxes), default=0.0)
            per = {c: max((b[1] for b in boxes if b[0] == i), default=0.0) for i, c in enumerate(CLASSES)}
            out.append(dict(p_problem=top, flag_problem=top >= self.conf, p=per,
                            flag={c: v >= self.conf for c, v in per.items()},
                            boxes=[[b[0], round(b[1], 4), *np.round(b[2:], 5).tolist()] for b in boxes]))
        return out
