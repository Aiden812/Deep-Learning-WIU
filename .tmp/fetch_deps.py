"""Fetch wheels from PyPI and unpack them into a target directory.

pip cannot run inside the DSH workspace-write sandbox: pip's scratch dirs come
from tempfile.mkdtemp, and those mkdir calls are denied (Errno 13) everywhere,
including inside the workspace. This script avoids temp dirs entirely by
extracting each wheel zip straight from memory.

Usage:
  python fetch_deps.py --target <dir> "name" "name==version" ...
  python fetch_deps.py --target <dir> --pair pydantic pydantic-core
      # --pair: pick the newest release of the first package whose exact pin
      # of the second package has a win_amd64 cp3xx wheel usable by this
      # interpreter (prefers cp314/abi3 over cp313).
"""
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def pick_win_wheel(files):
    """Best win_amd64 wheel for this interpreter: exact cp tag, then abi3, then cp313."""
    exact, abi3, older = [], [], []
    for f in files:
        fn = f["filename"]
        if not fn.endswith("win_amd64.whl"):
            continue
        if f"-{PY}-" in fn:
            exact.append(f)
        elif "abi3" in fn:
            abi3.append(f)
        else:
            older.append(f)
    for bucket in (exact, abi3, older):
        if bucket:
            return bucket[0]
    return None


def pick_pure_wheel(files):
    for f in files:
        if f["filename"].endswith("none-any.whl"):
            return f
    return None


def extract(url, target):
    with urllib.request.urlopen(url, timeout=180) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(target)


def install(name, target, version=None):
    url = f"https://pypi.org/pypi/{name}" + (f"/{version}" if version else "") + "/json"
    j = get_json(url)
    files = j["urls"] if "urls" in j else j["releases"][j["info"]["version"]]
    f = pick_pure_wheel(files) or pick_win_wheel(files)
    if f is None:
        raise SystemExit(f"{name} {version or j['info']['version']}: no usable wheel")
    print(f"{name} {version or j['info']['version']}: {f['filename']}")
    extract(f["url"], target)


def install_pair(dep_name, pin_name, target):
    """Newest dep_name whose exact pin of pin_name has a usable win wheel."""
    releases = get_json(f"https://pypi.org/pypi/{dep_name}/json")["releases"]
    def key(v):
        return [int(x) for x in re.findall(r"\d+", v)]
    for v in sorted(releases, key=key, reverse=True):
        try:
            info = get_json(f"https://pypi.org/pypi/{dep_name}/{v}/json")["info"]
        except Exception:
            continue
        pin = None
        for d in info.get("requires_dist") or []:
            m = re.match(rf"^{re.escape(pin_name)}==([0-9A-Za-z.]+)", d.split(";")[0].strip())
            if m:
                pin = m.group(1)
        if pin is None:
            continue
        try:
            files = get_json(f"https://pypi.org/pypi/{pin_name}/{pin}/json")["urls"]
        except Exception:
            continue
        wf = pick_win_wheel(files)
        pf = pick_pure_wheel(files)
        if not (wf or pf):
            continue
        pure = pick_pure_wheel(get_json(f"https://pypi.org/pypi/{dep_name}/{v}/json")["urls"])
        if pure is None:
            continue
        print(f"{dep_name} {v} -> {pin_name}=={pin}")
        extract(pure["url"], target)
        chosen = wf or pf
        print(f"{pin_name} {pin}: {chosen['filename']}")
        extract(chosen["url"], target)
        return
    raise SystemExit(f"no {dep_name} release with usable {pin_name} wheel")


def main():
    args = sys.argv[1:]
    target = os.path.abspath(os.path.join(HERE, "..", ".venv", "Lib", "site-packages"))
    if args[:1] == ["--target"]:
        target = os.path.abspath(args[1])
        args = args[2:]
    os.makedirs(target, exist_ok=True)
    if args[:1] == ["--pair"]:
        install_pair(args[1], args[2], target)
        args = args[3:]
    for spec in args:
        m = re.match(r"^([A-Za-z0-9._-]+)(?:==([0-9A-Za-z.]+))?$", spec)
        if not m:
            raise SystemExit(f"bad spec {spec!r}")
        install(m.group(1), target, m.group(2))
    print("done ->", target)


main()
