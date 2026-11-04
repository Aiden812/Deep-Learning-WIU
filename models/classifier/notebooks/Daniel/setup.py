import os
from pathlib import Path

# Store the dataset under data/raw/kaggle_oral/ (repo root is 4 levels above this script)
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "data" / "raw" / "kaggle_oral"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# kagglehub honors KAGGLEHUB_CACHE and nests downloads as <cache>/<dataset>/versions/<n>
os.environ["KAGGLEHUB_CACHE"] = str(DATASET_DIR)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("jhirshaikh/oral-image-dataset")

print("Path to dataset files:", path)
