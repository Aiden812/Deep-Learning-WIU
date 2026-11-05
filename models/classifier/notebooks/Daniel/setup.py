"""Download every raw dataset used by oral_condition_classifier.ipynb into data/raw/.

Layout produced (matches the paths the notebook reads):

    data/raw/kaggle_oral/datasets/jhirshaikh/oral-image-dataset/versions/<n>/...
    data/raw/zenodo/Benchmarking Dataset/{train,valid,test}/{images,yolo,coco,...}
    data/raw/zenodo/Dataset/{Annotations,Images}/...
    data/raw/dentalai/{train,valid,test}/{ann,img}   (DatasetNinja / Supervisely format)
    data/raw/mendeley_noncaries/...

Usage:
    python setup.py                         # everything
    python setup.py dentalai zenodo_benchmark   # only some datasets
    python setup.py --force                 # re-download even if already present
    python setup.py --keep-archives         # keep the .zip/.tar after extracting

kaggle_oral needs Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json).
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

# Repo root is 4 levels above this script
REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "raw"

USER_AGENT = "Deep-Learning-WIU-dataset-setup/1.0"

# Direct-download datasets. `wrap` = folder the archive's contents must end up in
# (inside `dir`); used when the archive may or may not already contain that top folder.
DATASETS = {
    "zenodo_benchmark": dict(
        dir="zenodo",
        wrap="Benchmarking Dataset",
        url="https://zenodo.org/records/14827784/files/Benchmarking%20Dataset.zip?download=1",
        filename="Benchmarking Dataset.zip",
        md5="852520f16faf65f4704ee06bd82561fa",
        source="https://zenodo.org/records/14827784",
    ),
    "zenodo_full": dict(
        dir="zenodo",
        wrap="Dataset",
        url="https://zenodo.org/records/14827784/files/Dataset.zip?download=1",
        filename="Dataset.zip",
        md5="89307871ac2f08f4a8f3a7da1f18db31",
        source="https://zenodo.org/records/14827784",
    ),
    "dentalai": dict(
        dir="dentalai",
        # Signed link from the "Download" button on https://datasetninja.com/dentalai
        url=(
            "https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMv"
            "MjA4NF9EZW50YWxhaS9kZW50YWxhaS1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJybmQ4ZG5hUXRtTjBX"
            "MUw2TndsUHBFLzhFd3Z1NEFBZUtFV3VCQ0RMcTFzPSJ9"
            "?response-content-disposition=attachment%3B%20filename%3D%22dentalai-DatasetNinja.tar%22"
        ),
        filename="dentalai-DatasetNinja.tar",
        source="https://datasetninja.com/dentalai",
    ),
    "mendeley_noncaries": dict(
        dir="mendeley_noncaries",
        url=(
            "https://data.mendeley.com/public-files/datasets/6zsnhrds9t/files/"
            "90a41c07-1193-4716-8fa1-1f3aeb3a0ff4/file_downloaded"
        ),
        filename="Mendeley dataset 19.4.2024.zip",
        sha256="b4aba842ba46349af92d557848deb85decbaf94a3e1a182a41a417f96025ea1b",
        source="https://data.mendeley.com/datasets/6zsnhrds9t/1",
    ),
}

ALL_NAMES = ["kaggle_oral", *DATASETS]


# --------------------------------------------------------------------------- helpers
def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, spec: dict) -> bool:
    for algo in ("sha256", "md5"):
        if algo in spec:
            ok = file_hash(path, algo) == spec[algo]
            print(f"  {algo} {'OK' if ok else 'MISMATCH'}")
            return ok
    return True  # no checksum published


def download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            if bar:
                bar.update(len(chunk))
        if bar:
            bar.close()
    tmp.replace(dest)


def extract(archive: Path, target: Path, wrap: str | None) -> None:
    """Extract `archive` into `target`. If `wrap` is set, contents land in target/wrap,
    whether or not the archive already has `wrap/` as its top folder."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.strip("/")]
            dest = target
            if wrap and not all(n.startswith(wrap + "/") for n in names):
                dest = target / wrap
            print(f"  extracting {len(names)} entries -> {dest}")
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            dest = target / wrap if wrap else target
            print(f"  extracting -> {dest}")
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:
                t.extractall(dest)
    else:
        raise RuntimeError(f"{archive} is neither a zip nor a tar file")


# --------------------------------------------------------------------------- datasets
def get_kaggle_oral(force: bool) -> None:
    dataset_dir = RAW_DIR / "kaggle_oral"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # kagglehub honors KAGGLEHUB_CACHE and nests downloads as <cache>/datasets/<owner>/<slug>/versions/<n>
    os.environ["KAGGLEHUB_CACHE"] = str(dataset_dir)
    import kagglehub

    path = kagglehub.dataset_download("jhirshaikh/oral-image-dataset", force_download=force)
    print(f"  -> {path}")


def get_direct(name: str, force: bool, keep_archives: bool) -> None:
    spec = DATASETS[name]
    target = RAW_DIR / spec["dir"]
    target.mkdir(parents=True, exist_ok=True)
    marker = target / f".{name}.complete"
    if marker.exists() and not force:
        print(f"  already present ({marker.name}); use --force to redo")
        return

    archive = target / spec["filename"]
    if archive.exists() and not force and verify(archive, spec):
        print(f"  reusing {archive.name}")
    else:
        print(f"  downloading {spec['source']}")
        download(spec["url"], archive)
        if not verify(archive, spec):
            raise RuntimeError(f"checksum mismatch for {archive}; delete it and retry")

    extract(archive, target, spec.get("wrap"))
    marker.write_text(spec["source"] + "\n")
    if not keep_archives:
        archive.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"subset to fetch (default: all). Choices: {', '.join(ALL_NAMES)}")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--keep-archives", action="store_true", help="keep .zip/.tar files after extraction")
    args = parser.parse_args()
    unknown = sorted(set(args.datasets) - set(ALL_NAMES))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}. Choices: {', '.join(ALL_NAMES)}")

    failed = []
    for name in args.datasets or ALL_NAMES:
        print(f"[{name}]")
        try:
            if name == "kaggle_oral":
                get_kaggle_oral(args.force)
            else:
                get_direct(name, args.force, args.keep_archives)
        except Exception as e:  # keep going so one bad link doesn't block the rest
            print(f"  FAILED: {e}")
            src = DATASETS.get(name, {}).get("source", "https://www.kaggle.com/datasets/jhirshaikh/oral-image-dataset")
            print(f"  download manually from {src}")
            failed.append(name)

    free = shutil.disk_usage(RAW_DIR).free / 1e9
    print(f"\nDone. {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}. ({free:.1f} GB free)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
