"""Write patient-level train/val/test splits of the Zenodo caries data in YOLO layout.

Outputs:
    data/processed/yolo/{images,labels}/{train,val,test}/   (D and d merged into class 0)
    manifests/yolo_split.csv   (key, patient_id, group, split)

Usage:
    python models/yolo_caries/scripts/Junwei/dedup_and_split.py --seed 42
"""

# shortcut: no dedup yet; 01_eda.ipynb found no duplicates in Zenodo. Add perceptual-hash dedup with a second source.

import argparse
import csv
import random
import re
import shutil
from collections import Counter
from pathlib import Path

TOL = 0.01  # allow 1% rounding at image edges, same as 01_eda.ipynb
SESSION_GAP_MIN = 30  # one person's two sessions (00/01) are photographed minutes apart under neighbouring IDs
LONER_GAP_MIN = 180  # an ID with no partner joins the nearest opposite-session ID photographed within 3 hours

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/zenodo_full"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/yolo"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/yolo_split.csv"))
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def patient_id(name: str) -> str | None:
    """Return the patient ID in a Zenodo filename, e.g. '003-007-1214', or None if it has none."""
    m = re.search(r"anonymous_(\d+[-_]\d+[-_]\d+)", name)
    if m:
        return m.group(1).replace('_','-')
    else:
        return None



def collect_pairs(src: Path) -> list[dict]:
    """Pair each YOLO label with its image by relative path without extension."""
    label_root, image_root = src / "Annotations" / "Darknet_YOLO", src / "Images"
    labels = {p.relative_to(label_root).with_suffix("").as_posix(): p for p in label_root.rglob("*.txt")}
    images = {p.relative_to(image_root).with_suffix("").as_posix(): p for p in image_root.rglob("*.jpg")}
    keys = labels.keys() & images.keys()
    print(f"{len(keys)} pairs, {len(labels.keys() - keys)} labels without image, "
          f"{len(images.keys() - keys)} images without label")

    records = []
    for key in sorted(keys):  # sorted so the seeded split is the same on every machine
        is_pilot = key.split("/")[0] == "pilot"
        pid = None if is_pilot else patient_id(Path(key).name)
        if pid is None and not is_pilot:
            raise ValueError(f"No patient ID in {key}")
        records.append({"key": key, "image_path": images[key], "label_path": labels[key],
                        "patient_id": pid, "is_pilot": is_pilot})
    return records


def has_bad_box(label_path: Path) -> bool:
    """True if any box in the label file has an edge outside [-TOL, 1 + TOL]."""
    for line in label_path.read_text().splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split())
            if x - w / 2 < -TOL or x + w / 2 > 1 + TOL or y - h / 2 < -TOL or y + h / 2 > 1 + TOL:
                return True
    return False


def id_info(records: list[dict]) -> dict[str, dict]:
    """Per patient ID: its session code and the timestamps (ms) of its photos. Pilot photos have no ID."""
    info = {}
    for r in records:
        if r["is_pilot"]:
            continue
        m = re.search(r"anonymous_\d+[-_]\d+[-_]\d+[-_](\d+)_(\d{13})", r["key"])
        if not m:
            raise ValueError(f"No session/timestamp in {r['key']}")
        info.setdefault(r["patient_id"], {"session": m.group(1), "times": []})["times"].append(int(m.group(2)))
    return info


def minutes_apart(a: dict, b: dict) -> float:
    return min(abs(s - t) for s in a["times"] for t in b["times"]) / 60_000


def person_groups(records: list[dict]) -> dict[str, str]:
    """Map each patient ID to a person group.

    Zenodo gives one person a new ID per session: e.g. 003-103-297 (session 01) and 003-103-298 (session 00),
    photographed minutes apart. IDs with opposite sessions, numbers <= 2 apart and photos within
    SESSION_GAP_MIN minutes are linked. Ambiguous neighbours end up in one group, which never leaks.
    """
    info = id_info(records)
    group = {p: p for p in info}

    def root(p: str) -> str:
        while group[p] != p:
            p = group[p]
        return p

    ids = sorted(info)
    for i, p in enumerate(ids):
        p_pre, p_num = p.rsplit("-", 1)
        for q in ids[i + 1:]:
            q_pre, q_num = q.rsplit("-", 1)
            if (p_pre == q_pre and abs(int(p_num) - int(q_num)) <= 2
                    and info[p]["session"] != info[q]["session"]
                    and minutes_apart(info[p], info[q]) <= SESSION_GAP_MIN):
                group[root(p)] = root(q)
    return {p: root(p) for p in info}


def adopt_loners(records: list[dict], group_of: dict[str, str]) -> dict[str, str]:
    """Put each ID that is alone in its group into the group of its nearest opposite-session ID.

    A loner may be the second session of someone the strict rule missed (e.g. numbers 3 apart, or 35 minutes).
    It joins the nearest ID with the same prefix and the other session code if that is within LONER_GAP_MIN.
    Joining a stranger's group costs nothing; leaving a real pair split would leak.
    """
    info = id_info(records)
    size = Counter(group_of.values())
    group = dict(group_of)

    def root(p: str) -> str:
        while group[p] != p:
            p = group[p]
        return p

    for p in sorted(p for p in group_of if size[group_of[p]] == 1):
        near = [(minutes_apart(info[p], info[q]), q) for q in info
                if q.rsplit("-", 1)[0] == p.rsplit("-", 1)[0] and info[q]["session"] != info[p]["session"]]
        if near and min(near)[0] <= LONER_GAP_MIN:
            group[p] = root(min(near)[1])  # p is still its own root here, so this cannot make a loop
    return {p: root(p) for p in group}


def assign_splits(records: list[dict], ratios: list[float], seed: int) -> None:
    """Shuffle person groups (not photos or IDs), cut by ratios, and set r["group"] and r["split"].

    Pilot photos have no ID, so they all go to train, and so does anyone photographed while the pilot was running.
    """
    group_of = person_groups(records)
    groups = sorted(set(group_of.values()))  # shuffled before loners are adopted, so adopting moves only the loners
    random.Random(seed).shuffle(groups)
    cut1 = int(ratios[0] * len(groups))
    cut2 = int((ratios[0] + ratios[1]) * len(groups))
    split_of = {g: "train" for g in groups[:cut1]}
    split_of.update({g: "val" for g in groups[cut1:cut2]})
    split_of.update({g: "test" for g in groups[cut2:]})

    group_of = adopt_loners(records, group_of)
    pilot_end = max((int(re.search(r"\d{13}", r["key"]).group()) for r in records if r["is_pilot"]), default=0)
    for p, i in id_info(records).items():
        if min(i["times"]) <= pilot_end:  # could be a pilot person photographed again under an ID
            split_of[group_of[p]] = "train"

    for r in records:
        r["group"] = group_of.get(r["patient_id"])
        r["split"] = "train" if r["is_pilot"] else split_of[r["group"]]


def write_split(records: list[dict], dst: Path, manifest: Path) -> None:
    """Copy images, write labels with every class as 0 (caries), and save the manifest."""
    for kind in ("images", "labels"):
        for s in ("train", "val", "test"):
            folder = dst / kind / s
            folder.mkdir(parents=True, exist_ok=True)
            for old in [*folder.glob("*.jpg"), *folder.glob("*.txt")]:  # clear a previous run, keep .gitkeep
                old.unlink()

    for r in records:
        name = r["key"].replace("/", "__")  # flat but still unique, e.g. no_retractors__Frontal__anonymous_...
        shutil.copy2(r["image_path"], dst / "images" / r["split"] / f"{name}.jpg")
        lines = [line.split() for line in r["label_path"].read_text().splitlines() if line.strip()]
        (dst / "labels" / r["split"] / f"{name}.txt").write_text("".join(f"0 {' '.join(p[1:])}\n" for p in lines))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "patient_id", "group", "split"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    assert abs(sum(args.ratios) - 1) < 1e-6, "--ratios must sum to 1, e.g. 0.7 0.15 0.15"
    records = collect_pairs(args.src)
    print(f"{sum(r['is_pilot'] for r in records)} pilot images, "
          f"{len({r['patient_id'] for r in records} - {None})} patients")

    kept = [r for r in records if not has_bad_box(r["label_path"])]
    print(f"{len(records) - len(kept)} dropped for boxes outside the image, {len(kept)} kept")
    records = kept

    assign_splits(records, args.ratios, args.seed)
    for s in ("train", "val", "test"):
        in_split = [r for r in records if r["split"] == s]
        print(f"{s}: {len(in_split)} images ({len(in_split) / len(records):.0%}), "
              f"{len({r['group'] for r in in_split} - {None})} person groups, "
              f"{len({r['patient_id'] for r in in_split} - {None})} patient IDs")

    write_split(records, args.dst, args.manifest)
    print(f"Wrote {len(records)} images and labels to {args.dst}, manifest to {args.manifest}")


if __name__ == "__main__":
    main()
