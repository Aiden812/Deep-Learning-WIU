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
    """Hex digest of `path` under `algo` ("md5"/"sha256").

    Reads in 1 MB chunks so multi-GB archives are never loaded into memory - these hashes are taken
    over the whole archive on every resume check, so the streaming cost matters.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, spec: dict) -> bool:
    """Check `path` against whichever checksum the dataset spec publishes; True if none is published.

    Checks sha256 first, then md5 - each upstream publishes at most one, and dentalai publishes none
    (its Supervisely link is signed, and re-signing would change nothing about the content, so there
    is nothing stable to pin). Returns instead of raising so the caller can decide whether a mismatch
    is fatal (fresh download) or just means "re-download" (leftover archive).
    """
    for algo in ("sha256", "md5"):
        if algo in spec:
            ok = file_hash(path, algo) == spec[algo]
            print(f"  {algo} {'OK' if ok else 'MISMATCH'}")
            return ok
    return True  # no checksum published


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` with an optional tqdm bar.

    Writes to a `.part` sibling first and renames only on success (os.replace is atomic on the same
    volume), so an interrupted or failed download never looks like a complete file to the later
    checksum/extract steps - a stale `.part` is simply overwritten on the next attempt. The User-Agent
    header is required by some hosts (Zenodo rejects the default Python UA). 60 s is a per-read socket
    timeout, not a whole-download deadline, so slow large files are fine.
    """
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
    """Extract `archive` (zip or tar) into `target`.

    `wrap` names the folder the contents must end up in (inside `target`) because the notebook expects
    e.g. `zenodo/Benchmarking Dataset/train/images`. Zenodo zips are inconsistent about whether the
    uploaded archive already contains that top folder, so for zips we only add the wrapper when it is
    missing (checked by testing whether every entry starts with `wrap/`); tars from these sources never
    contain it, so they always extract into `target/wrap`.

    tar extraction uses the `data` filter (Python >= 3.12) which refuses path-traversal members
    (`../...`) and odd file types; on older Pythons we extract unfiltered, same as `tar` CLI would.
    """
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
    """Fetch kaggle_oral via kagglehub into data/raw/kaggle_oral (credentials required).

    kagglehub ignores its default cache location here: pointing KAGGLEHUB_CACHE at data/raw/kaggle_oral
    makes it nest the download as <cache>/datasets/<owner>/<slug>/versions/<n>/, which is exactly the
    layout the notebook's find_dataset() searches. Kaggle needs KAGGLE_USERNAME/KAGGLE_KEY or
    ~/.kaggle/kaggle.json; without them this raises and main() reports the manual-download URL instead.
    """
    dataset_dir = RAW_DIR / "kaggle_oral"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # kagglehub honors KAGGLEHUB_CACHE and nests downloads as <cache>/datasets/<owner>/<slug>/versions/<n>
    os.environ["KAGGLEHUB_CACHE"] = str(dataset_dir)
    import kagglehub

    path = kagglehub.dataset_download("jhirshaikh/oral-image-dataset", force_download=force)
    print(f"  -> {path}")


def get_direct(name: str, force: bool, keep_archives: bool) -> None:
    """Download, checksum, and extract one direct-URL dataset from DATASETS.

    Three-step resume, cheapest check first, so re-running setup.py after an interruption is fast and
    never touches a good download:
      1. `.<name>.complete` marker in the target dir -> the whole pipeline finished once: skip
         everything (only --force redoes it).
      2. archive already present AND its checksum still matches -> skip the download, re-extract only.
         (A half-written archive fails the checksum and is re-downloaded.)
      3. otherwise download, then verify - a mismatch after a fresh download is fatal: the URL or the
         published hash changed, and extracting a corrupt archive would poison data/raw.

    After success the marker is written (holding the source URL as provenance) and the archive is
    deleted unless --keep-archives.
    """
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
    """CLI entry: fetch the requested datasets (default all), returning 1 if any failed.

    Datasets are independent, so a failure (expired Supervisely link, no Kaggle credentials, disk
    full) is caught per dataset: it prints the manual-download URL and continues with the rest. The
    exit code is 1 iff at least one failed, which is what a CI/agent wrapper would check; the
    remaining free-disk line at the end is the usual culprit when downloads suddenly start failing.
    """
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
