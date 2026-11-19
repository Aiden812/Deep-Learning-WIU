"""Paths and constants shared by the benchmark scripts."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # Deep-Learning-WIU
WORKSPACE = REPO.parent                             # folder that also holds datasets/ and .venv-train/
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "processed" / "benchmark"     # images, labels, predictions (gitignored)
PRED_DIR = OUT / "predictions"
MANIFEST = Path(__file__).resolve().parent / "manifest.csv"   # committed: the frozen benchmark
RESULTS = Path(__file__).resolve().parent / "results"

# Our 5 condition ids, as in oral_datasetv1.2/pool/classes.txt and the YOLO models
CLASSES = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]
# manifest column holding the photo-level truth for each class (1 / 0 / empty = unknown)
CLASS_COL = {"Calculus": "calculus", "Caries": "caries", "Hypodontia": "missing_teeth",
             "Mouth_Ulcer": "ulcer", "Tooth_Discoloration": "discoloration"}
SEED = 42
COPY_DIST = 6          # pHash distance (of 64 bits) at or under which a photo counts as a copy of a training photo
                       # (eye-checked: copies were <= 6; at 8 they were different photos with the same framing)
SIFT_COPY = 25         # matching keypoints after a geometric fit at or above which two photos are the same photo
