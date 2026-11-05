"""Scrape dental condition images from Bing, Baidu and DuckDuckGo.

Google is not included: it now serves a JavaScript-only page to scripts and a
CAPTCHA to headless browsers, so it cannot be scraped without bypassing its bot
protection. (DuckDuckGo/Bing results overlap heavily with Google's anyway.)

Setup:  pip install -r models/classifier/scripts/Daniel/requirements.txt

Usage:
    python models/classifier/scripts/Daniel/scrape_web_images.py
    python models/classifier/scripts/Daniel/scrape_web_images.py --classes Caries Mouth_Ulcer --target 2000
    python models/classifier/scripts/Daniel/scrape_web_images.py --engines bing ddg --threads 16

Writes:
    <out>/<Class>/<md5>.<ext>   one folder per class, same names as prepare_classifier_data.py
    <out>/manifest.csv          file, class, engine, query, source URL (for attribution/cleaning)
    <out>/_progress.json        (engine, query) pairs already finished, so a re-run resumes

One search query gives at most ~1000 results per engine, so each class is
searched with many query variants across several engines until it reaches
--target unique images. Exact duplicates (same bytes) are dropped; images that
are too small, broken or GIFs are dropped. Near-duplicates and off-topic images
(cartoons, X-rays, diagrams, product shots) are NOT removed here -- run
Aiden/filter_images.py and do the human review step before training.

Rare conditions (e.g. Hypodontia) will run out of unique images on the web well
before 15000; the script stops for a class once every query is exhausted.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from icrawler import ImageDownloader
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler

try:
    from ddgs import DDGS
except ImportError:  # older package name
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO / "data" / "raw" / "external"

TARGET = 15000        # unique images wanted per class
PER_QUERY = 1000      # most results one engine will return for one query
MIN_SIDE = 128        # same threshold as filter_images.py
FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp"}  # GIFs are mostly clip art
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

# Class folder name -> search queries, most relevant first.
# Folder names match models/classifier/scripts/Aiden/prepare_classifier_data.py.
QUERIES = {
    "Normal": [
        "teeth", "healthy teeth", "healthy teeth close up", "normal teeth intraoral photo",
        "healthy gums and teeth", "teeth smile close up", "white teeth close up", "open mouth teeth",
        "straight teeth", "clean teeth after dental cleaning", "healthy dentition intraoral photograph",
        "healthy child teeth", "adult teeth close up", "upper front teeth", "lower front teeth",
        "molars close up", "healthy smile",
    ],
    "Calculus": [
        "calculus teeth", "dental calculus", "tartar on teeth", "tartar buildup teeth",
        "dental tartar close up", "plaque and tartar teeth", "calculus on lower front teeth",
        "supragingival calculus", "subgingival calculus", "heavy calculus teeth",
        "tartar teeth before scaling", "yellow tartar teeth", "calcified plaque teeth",
        "tartar along gum line",
    ],
    "Caries": [
        "caries", "dental caries", "tooth decay", "cavities teeth", "tooth cavity close up",
        "decayed teeth", "rotten teeth", "carious lesion tooth", "early childhood caries",
        "baby bottle tooth decay", "molar cavity", "occlusal caries", "interproximal caries photo",
        "black spots on teeth cavity", "rampant caries", "cervical caries", "brown spot tooth decay",
    ],
    "Gingivitis": [
        "gingivitis", "swollen gums", "red inflamed gums", "bleeding gums", "gum disease",
        "gingival inflammation", "periodontitis gums", "plaque induced gingivitis",
        "gum infection teeth", "receding gums", "puffy gums", "gingival hyperplasia",
    ],
    "Hypodontia": [
        "hypodontia", "congenitally missing teeth", "congenitally missing lateral incisors",
        "tooth agenesis", "missing lateral incisor", "oligodontia", "anodontia",
        "missing tooth gap", "missing teeth smile", "peg lateral incisor", "missing premolar",
        "ectodermal dysplasia teeth", "diastema missing tooth",
    ],
    "Mouth_Ulcer": [
        "mouth ulcer", "canker sore", "aphthous ulcer", "aphthous stomatitis", "mouth sore",
        "ulcer on gum", "ulcer on tongue", "ulcer inside lip", "oral ulcer",
        "recurrent aphthous ulcer", "traumatic ulcer mouth", "herpetic stomatitis",
        "mouth ulcer inner cheek",
    ],
    "Tooth_Discoloration": [
        "tooth discoloration", "stained teeth", "yellow teeth", "brown stains on teeth",
        "tetracycline stained teeth", "dental fluorosis", "grey tooth", "dead tooth discoloration",
        "coffee stained teeth", "tobacco stained teeth", "smoker teeth", "white spots on teeth",
        "black stains on teeth", "enamel hypoplasia discoloration", "intrinsic tooth stain",
        "extrinsic tooth stain",
    ],
}
# Each base query is also tried with these endings, which pulls in different results.
SUFFIXES = ["", " close up", " photo", " clinical photo", " patient"]
ENGINES = ["bing", "ddg", "baidu"]

LOCK = threading.Lock()
log = logging.getLogger("scrape")


class ClassStore:
    """Saves validated, de-duplicated images for one class and logs them to the manifest."""

    def __init__(self, folder, cls, manifest, target, min_side):
        self.folder, self.cls, self.manifest = folder, cls, manifest
        self.target, self.min_side = target, min_side
        folder.mkdir(parents=True, exist_ok=True)
        # File names are content hashes, so what is already on disk is the duplicate set.
        self.hashes = {p.stem for p in folder.iterdir() if p.suffix.lstrip(".") in FORMATS.values()}

    @property
    def count(self):
        return len(self.hashes)

    @property
    def full(self):
        return self.count >= self.target

    def save(self, content, url, engine, query):
        """Write the image if it is new, valid and big enough. Returns True if saved."""
        digest = hashlib.md5(content).hexdigest()
        with LOCK:
            if digest in self.hashes or self.full:
                return False
        try:
            img = Image.open(io.BytesIO(content))
            fmt, (w, h) = img.format, img.size
            img.verify()  # catches truncated/corrupt files
        except Exception:
            return False
        if fmt not in FORMATS or min(w, h) < self.min_side:
            return False
        with LOCK:
            if digest in self.hashes or self.full:
                return False
            self.hashes.add(digest)
            name = f"{digest}.{FORMATS[fmt]}"
            (self.folder / name).write_bytes(content)
            self.manifest.writerow([f"{self.cls}/{name}", self.cls, engine, query, url, w, h])
        return True


class StoreDownloader(ImageDownloader):
    """icrawler downloader that hands every image to a ClassStore instead of saving it directly.
    store/engine/query are set as class attributes by make_downloader()."""
    store = engine = query = None

    def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
        task["success"], task["filename"] = False, None
        if self.store.full:
            self.signal.set(reach_max_num=True)
            return False
        for _ in range(max_retry):
            try:
                resp = self.session.get(task["file_url"], timeout=timeout)
            except Exception:
                continue
            if resp.status_code == 200 and self.store.save(resp.content, task["file_url"], self.engine, self.query):
                with self.lock:
                    self.fetched_num += 1
                task["success"] = True
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
            return task["success"]
        return False


def make_downloader(store, engine, query):
    return type("BoundDownloader", (StoreDownloader,), {"store": store, "engine": engine, "query": query})


def crawl_icrawler(engine, query, store, max_num, threads):
    crawler_cls = {"bing": BingImageCrawler, "baidu": BaiduImageCrawler}[engine]
    crawler = crawler_cls(
        downloader_cls=make_downloader(store, engine, query),
        parser_threads=2,
        downloader_threads=threads,
        storage={"root_dir": str(store.folder)},  # required by icrawler; StoreDownloader does the writing
        log_level=logging.CRITICAL,
    )
    kwargs = {"keyword": query, "max_num": max_num}
    if engine == "bing":
        kwargs["filters"] = {"type": "photo"}
    crawler.crawl(**kwargs, max_idle_time=60)


def crawl_ddg(query, store, max_num, threads):
    if DDGS is None:
        raise RuntimeError("pip install ddgs")
    results = DDGS().images(query, max_results=max_num, type_image="photo")
    urls = [r["image"] for r in results if r.get("image")]
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = requests.adapters.HTTPAdapter(pool_connections=threads, pool_maxsize=threads)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def fetch(url):
        if store.full:
            return
        try:
            resp = session.get(url, timeout=10)
        except Exception:
            return
        if resp.status_code == 200:
            store.save(resp.content, url, "ddg", query)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(fetch, urls))


def load_progress(path):
    try:
        return {k: set(v) for k, v in json.loads(path.read_text()).items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_progress(path, progress):
    path.write_text(json.dumps({k: sorted(v) for k, v in progress.items()}, indent=1))


def scrape_class(cls, args, manifest, progress, progress_path):
    store = ClassStore(args.out / cls, cls, manifest, args.target, args.min_side)
    done = progress.setdefault(cls, set())
    queries = [q + s for q in QUERIES[cls] for s in SUFFIXES]
    log.info("== %s: %d on disk, target %d, %d queries x %d engines",
             cls, store.count, args.target, len(queries), len(args.engines))

    for query in queries:
        for engine in args.engines:
            if store.full:
                log.info("== %s: reached %d images", cls, store.count)
                return store.count
            key = f"{engine}|{query}"
            if key in done:
                continue
            before = store.count
            try:
                if engine == "ddg":
                    crawl_ddg(query, store, args.per_query, args.threads)
                else:
                    crawl_icrawler(engine, query, store, min(args.per_query, args.target - store.count), args.threads)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # one engine breaking (rate limit, HTML change) shouldn't stop the run
                log.warning("   %-6s %-45s failed: %s", engine, query, e)
                time.sleep(args.delay * 5)
                continue
            log.info("   %-6s %-45s +%-5d total %d", engine, query, store.count - before, store.count)
            done.add(key)
            with LOCK:
                save_progress(progress_path, progress)
            time.sleep(args.delay + random.random() * args.delay)

    log.info("== %s: queries exhausted at %d images (target %d)", cls, store.count, args.target)
    return store.count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default data/raw/external)")
    ap.add_argument("--classes", nargs="+", default=list(QUERIES), choices=list(QUERIES))
    ap.add_argument("--engines", nargs="+", default=ENGINES, choices=ENGINES)
    ap.add_argument("--target", type=int, default=TARGET, help="unique images per class")
    ap.add_argument("--per-query", type=int, default=PER_QUERY, help="max results per engine per query")
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    ap.add_argument("--threads", type=int, default=8, help="download threads")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to pause between queries")
    args = ap.parse_args()

    # Root at WARNING keeps icrawler's per-thread INFO chatter quiet; our own logger stays at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "_progress.json"
    progress = load_progress(progress_path)
    manifest_path = args.out / "manifest.csv"
    new_manifest = not manifest_path.exists()

    totals = {}
    with open(manifest_path, "a", newline="", encoding="utf-8", buffering=1) as f:
        manifest = csv.writer(f)
        if new_manifest:
            manifest.writerow(["file", "class", "engine", "query", "url", "width", "height"])
        try:
            for cls in args.classes:
                totals[cls] = scrape_class(cls, args, manifest, progress, progress_path)
        except KeyboardInterrupt:
            log.info("Interrupted -- progress saved, re-run the same command to resume.")
        finally:
            save_progress(progress_path, progress)

    log.info("Summary:")
    for cls, n in totals.items():
        log.info("  %-20s %6d", cls, n)


if __name__ == "__main__":
    sys.exit(main())
