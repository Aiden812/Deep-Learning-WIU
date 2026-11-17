"""One adapter per kind of model. Each turns a PIL photo into the same output:

    {"p_problem": float | None,          # chance the photo shows any of our 5 conditions
     "flag_problem": bool | None,        # the model's own verdict at its own saved threshold
     "p": {class: float},                # per-condition score (only models that name conditions)
     "flag": {class: bool},
     "boxes": [[class_id, conf, x1, y1, x2, y2], ...]}   # detectors only, normalised 0-1 coordinates

Every adapter preprocesses exactly as its model was trained/served. Nothing here is tuned on the benchmark.
To add a model: add an entry to MODELS (weights path relative to the repo) with the right kind.
"""
from pathlib import Path

from common import CLASSES, REPO

# kind -> loader in this package. "python" = which interpreter run.py must use (keras lives in the repo's .venv)
MODELS = {
    # ---- Brandon ----------------------------------------------------------------------------------------------
    "brandon_cls_v12": dict(kind="torchvision_cls", weights="models/classifier/weights/Brandon/issue_classifier_v12.pt",
                            owner="Brandon", note="EfficientNet-B0 gate, v1.2 data"),
    "brandon_cls_v11": dict(kind="torchvision_cls", weights="models/classifier/weights/Brandon/issue_classifier.pt",
                            owner="Brandon", note="EfficientNet-B0 gate, v1.1 data"),
    "brandon_yolo_v12": dict(kind="yolo", weights="models/yolo_caries/weights/Brandon/yolo_detector_v12.pt",
                             owner="Brandon", note="YOLO11n 640, v1.2 data", conf=0.25),
    "brandon_yolo_v11": dict(kind="yolo", weights="models/yolo_caries/weights/Brandon/yolo_detector.pt",
                             owner="Brandon", note="YOLO11n 640, v1.1 data", conf=0.25),
    # ---- team baseline (Keras EfficientNetB0, 6 classes incl. Normal) ------------------------------------------
    "team_keras_6class": dict(kind="keras6", weights="models/classifier/weights/oral_classifier.keras",
                              owner="team", note="Keras EfficientNetB0 6-class baseline", python="repo_venv"),
    # ---- Aiden (drop the weights at these paths) ---------------------------------------------------------------
    "aiden_gate_dinov2": dict(kind="aiden_gate", weights="models/classifier/results/Aiden/dinov2_vitb14/model.pt",
                              owner="Aiden", note="DINOv2 ViT-B/14 gate"),
    "aiden_gate_densenet": dict(kind="aiden_gate", weights="models/classifier/results/Aiden/densenet121/model.pt",
                                owner="Aiden", note="DenseNet-121 gate"),
    "aiden_yolo11n": dict(kind="yolo", weights="models/yolo_caries/weights/Aiden/yolo11n.pt", owner="Aiden",
                          note="YOLO11n on Daniel's v4 data", conf=0.25),
    "aiden_yolo11s": dict(kind="yolo", weights="models/yolo_caries/weights/Aiden/yolo11s.pt", owner="Aiden",
                          note="YOLO11s on Daniel's v4 data", conf=0.25),
    # ---- Daniel ------------------------------------------------------------------------------------------------
    "daniel_cls_binary": dict(kind="torchvision_cls", weights="models/classifier/weights/Daniel/binary/best.pt",
                              owner="Daniel", note="binary gate"),
    "daniel_cls_6class": dict(kind="torchvision_cls", weights="models/classifier/weights/Daniel/best.pt",
                              owner="Daniel", note="6-class condition classifier"),
    "daniel_yolo": dict(kind="yolo", weights="models/yolo_caries/weights/Daniel/best.pt", owner="Daniel",
                        note="YOLO (conf from best.json sidecar)"),
}


def norm_name(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


# names other models use for our 5 classes (lower-case, spaces -> _)
ALIASES = {"calculus": "Calculus", "caries": "Caries", "dental_caries": "Caries", "hypodontia": "Hypodontia",
           "missing_teeth": "Hypodontia", "mouth_ulcer": "Mouth_Ulcer", "ulcer": "Mouth_Ulcer",
           "tooth_discoloration": "Tooth_Discoloration", "discoloration": "Tooth_Discoloration"}
HEALTHY_NAMES = {"normal", "healthy", "no_issue", "none"}


def to_ours(name: str):
    return ALIASES.get(norm_name(name))


def load(name: str, weights: str | None = None):
    spec = dict(MODELS[name])
    w = Path(weights) if weights else REPO / spec["weights"]
    if not w.exists():
        raise FileNotFoundError(f"{name}: weights not found at {w}")
    kind = spec["kind"]
    if kind == "torchvision_cls":
        from .torchvision_cls import TorchvisionCls as C
    elif kind == "yolo":
        from .yolo import Yolo as C
    elif kind == "aiden_gate":
        from .aiden_gate import AidenGate as C
    elif kind == "keras6":
        from .keras6 import Keras6 as C
    else:
        raise ValueError(kind)
    return C(w, **{k: v for k, v in spec.items() if k == "conf"})


__all__ = ["MODELS", "CLASSES", "load", "to_ours", "HEALTHY_NAMES"]
