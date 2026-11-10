"""Scrape dental condition images from Baidu (Bing / DuckDuckGo optional, see below).

Baidu is searched with Chinese queries (QUERIES_ZH): English queries on Baidu
mostly return book covers and shop listings, while Chinese clinical terms
(e.g. 龋齿 口内照) return real intraoral photos.

Engines that don't work for scripts (checked Nov 2026):
  - Google: JavaScript-only page / CAPTCHA, can't be scraped without bypassing bot protection.
  - Bing:   serves random unrelated images (news photos, anime, products) to scripted
            requests, so everything it returns is off-topic. Off by default.
  - DuckDuckGo: image API rejects plain HTTP clients, and the ddgs library's image
            search actually queries Bing behind the scenes. Off by default.
Pass e.g. --engines baidu bing to try them anyway; a worker gives up on an engine
after several failed queries in a row.

Setup:  pip install -r models/classifier/scripts/Daniel/requirements.txt

Usage:
    python models/classifier/scripts/Daniel/scrape_web_images.py
    python models/classifier/scripts/Daniel/scrape_web_images.py --classes Caries Mouth_Ulcer --target 2000
    python models/classifier/scripts/Daniel/scrape_web_images.py --threads 16 --jobs 4

Writes:
    <out>/<Class>/<md5>.<ext>   one folder per class, same names as prepare_classifier_data.py
    <out>/manifest.csv          file, class, engine, query, source URL (for attribution/cleaning)
    <out>/_progress.json        (engine, query) pairs already finished, so a re-run resumes

Parallelism: every (class, engine) pair runs as its own worker, so all classes
are scraped at once (and, with several engines, each engine is queried at the
same time). --jobs caps how many queries hit the same engine at once (to avoid
rate limits), and --threads is the number of image-download threads per query.

One search query gives at most ~1000 results per engine, so each class is
searched with many query variants across several engines until it reaches
--target unique images. Exact duplicates (same bytes) are dropped; images that
are too small, broken or GIFs are dropped. Near-duplicates and off-topic images
(cartoons, X-rays, diagrams, product shots) are NOT removed here -- run
Aiden/filter_images.py and do the human review step before training.

Rare conditions (e.g. Hypodontia) may run out of unique images on the web before
reaching --target; the script stops for a class once every query is exhausted.
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
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

TARGET = 3000         # unique images wanted per class
PER_QUERY = 1000      # most results one engine will return for one query
MIN_SIDE = 128        # same threshold as filter_images.py
JOBS = 3              # queries running at once on the same engine
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

# Chinese queries, used for Baidu. Same classes as QUERIES.
QUERIES_ZH = {
    "Normal": [
        "健康牙齿", "正常牙齿 口内照", "健康牙龈", "牙齿特写", "整齐的牙齿", "儿童健康牙齿",
        "洁白的牙齿", "张嘴 牙齿", "健康 恒牙", "磨牙 特写",
    ],
    "Calculus": [
        "牙结石", "牙石", "牙结石 口内照", "下前牙 牙结石", "龈上牙石", "龈下牙石",
        "洗牙前 牙结石", "牙结石 很厚", "牙垢 牙结石",
    ],
    "Caries": [
        "龋齿", "蛀牙", "龋齿 口内照", "乳牙龋齿", "窝沟龋", "邻面龋", "猖獗龋",
        "牙齿龋坏", "奶瓶龋", "牙齿 龋洞",
    ],
    "Gingivitis": [
        "牙龈炎", "牙龈红肿", "牙龈出血", "牙周炎", "牙龈发炎", "牙龈增生", "牙龈萎缩",
        "牙龈炎 口内照",
    ],
    "Hypodontia": [
        "先天缺牙", "先天性缺牙", "牙齿发育不全 缺牙", "侧切牙先天缺失", "恒牙先天缺失",
        "过小牙 锥形牙", "外胚层发育不良 牙齿", "先天缺牙 口内照",
    ],
    "Mouth_Ulcer": [
        "口腔溃疡", "复发性口腔溃疡", "阿弗他溃疡", "舌头溃疡", "嘴唇内侧溃疡", "牙龈溃疡",
        "创伤性溃疡 口腔", "疱疹性口炎", "口腔溃疡 口内照",
    ],
    "Tooth_Discoloration": [
        "四环素牙", "氟斑牙", "牙齿变色", "牙齿发黄", "牙齿色素沉着", "烟渍牙", "死髓牙 变色",
        "牙齿黑斑", "牙釉质发育不全", "茶渍牙",
    ],
}
SUFFIXES_ZH = ["", " 口内照", " 图片", " 临床", " 病例"]

ENGINES = ["baidu", "bing", "ddg"]
DEFAULT_ENGINES = ["baidu"]
MAX_FAILS = 3  # consecutive failed queries before a worker gives up on its engine

LOCK = threading.Lock()       # guards stores, manifest and progress file
STOP = threading.Event()      # set on Ctrl+C so every worker winds down
log = logging.getLogger("scrape")


class ClassStore:
    """Saves validated, de-duplicated images for one class and logs them to the manifest.
    Shared by all engine workers of that class, so every mutation happens under LOCK."""

    def __init__(self, folder, cls, manifest, target, min_side):
        self.folder, self.cls, self.manifest = folder, cls, manifest
        self.target, self.min_side = target, min_side
        folder.mkdir(parents=True, exist_ok=True)
        # File names are content hashes, so what is already on disk is the duplicate set.
        self.hashes = {p.stem for p in folder.iterdir() if p.suffix.lstrip(".") in FORMATS.values()}
        self.added = Counter()  # "engine|query" -> images saved, for per-query log lines

    @property
    def count(self):
        return len(self.hashes)

    @property
    def full(self):
        return self.count >= self.target

    @property
    def stopped(self):
        """True when downloading for this class should stop (target hit or Ctrl+C)."""
        return STOP.is_set() or self.full

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
            self.added[f"{engine}|{query}"] += 1
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
        if self.store.stopped:
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
        if store.stopped:
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
    """Caller must hold LOCK (workers add to the sets concurrently)."""
    path.write_text(json.dumps({k: sorted(v) for k, v in progress.items()}, indent=1))


def scrape_worker(store, engine, args, progress, progress_path, engine_slots):
    """Run every query of one class on one engine. One of these runs per (class, engine) pair."""
    cls = store.cls
    with LOCK:
        done = progress.setdefault(cls, set())
    if engine == "baidu":
        queries = [q + s for q in QUERIES_ZH[cls] for s in SUFFIXES_ZH]
    else:
        queries = [q + s for q in QUERIES[cls] for s in SUFFIXES]
    fails = 0

    for query in queries:
        if store.stopped:
            return
        key = f"{engine}|{query}"
        if key in done:
            continue
        failed = None
        with engine_slots:  # at most --jobs queries per engine at once
            if store.stopped:
                return
            try:
                if engine == "ddg":
                    crawl_ddg(query, store, args.per_query, args.threads)
                else:
                    crawl_icrawler(engine, query, store,
                                   max(1, min(args.per_query, args.target - store.count)), args.threads)
            except Exception as e:  # one engine breaking (rate limit, HTML change) shouldn't stop the run
                failed = e
        if failed is not None:
            fails += 1
            log.warning("   %-19s %-6s %-45s failed: %s", cls, engine, query, failed)
            if fails >= MAX_FAILS:
                log.warning("== %s: giving up on %s after %d failures in a row", cls, engine, fails)
                return
            STOP.wait(args.delay * 5)
            continue
        fails = 0
        with LOCK:
            added = store.added.pop(key, 0)
            done.add(key)
            save_progress(progress_path, progress)
        log.info("   %-19s %-6s %-45s +%-5d total %d", cls, engine, query, added, store.count)
        STOP.wait(args.delay + random.random() * args.delay)  # returns early on Ctrl+C

    if not store.stopped:
        log.info("== %s: %s queries exhausted at %d images", cls, engine, store.count)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default data/raw/external)")
    ap.add_argument("--classes", nargs="+", default=list(QUERIES), choices=list(QUERIES))
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES, choices=ENGINES)
    ap.add_argument("--target", type=int, default=TARGET, help="unique images per class")
    ap.add_argument("--per-query", type=int, default=PER_QUERY, help="max results per engine per query")
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    ap.add_argument("--threads", type=int, default=8, help="download threads per query")
    ap.add_argument("--jobs", type=int, default=JOBS, help="queries running at once per engine")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds each worker pauses between queries")
    args = ap.parse_args()

    # Root at WARNING keeps icrawler's per-thread INFO chatter quiet; our own logger stays at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "_progress.json"
    progress = load_progress(progress_path)
    manifest_path = args.out / "manifest.csv"
    new_manifest = not manifest_path.exists()

    with open(manifest_path, "a", newline="", encoding="utf-8", buffering=1) as f:
        manifest = csv.writer(f)
        if new_manifest:
            manifest.writerow(["file", "class", "engine", "query", "url", "width", "height"])

        stores = {cls: ClassStore(args.out / cls, cls, manifest, args.target, args.min_side)
                  for cls in args.classes}
        for cls, store in stores.items():
            log.info("== %s: %d on disk, target %d", cls, store.count, args.target)
        engine_slots = {e: threading.BoundedSemaphore(args.jobs) for e in args.engines}

        with ThreadPoolExecutor(len(stores) * len(args.engines)) as pool:
            futures = [pool.submit(scrape_worker, store, engine, args, progress, progress_path, engine_slots[engine])
                       for store in stores.values() for engine in args.engines]
            try:
                # Short timeouts keep the main thread responsive to Ctrl+C (esp. on Windows).
                while wait(futures, timeout=1).not_done:
                    pass
            except KeyboardInterrupt:
                log.info("Interrupted -- letting in-flight queries wind down, then saving progress...")
                STOP.set()
        for fut in futures:
            if fut.done() and fut.exception():
                log.error("worker crashed: %r", fut.exception())

        with LOCK:
            save_progress(progress_path, progress)
    if STOP.is_set():
        log.info("Progress saved -- re-run the same command to resume.")

    log.info("Summary:")
    for cls, store in stores.items():
        log.info("  %-20s %6d", cls, store.count)


if __name__ == "__main__":
    sys.exit(main())
