"""Split the v1.2 pool into train / val / test (fills the `split` column of pool.csv).

Rules:
- A *unit* is a group (copies of one photo, v1.1 patient groups) merged with every other photo of the same
  Zenodo patient. A unit always goes to ONE split, so the exam never contains a mouth the model trained on.
- Each (source, healthy/problem) stratum is split on its own: 70 / 15 / 15. So every source that has both
  healthy and problem photos puts BOTH kinds into val and test. These mixed sources have few healthy photos,
  so their healthy stratum is split 50 / 25 / 25, which gives the fair exams enough healthy photos.
- `val_fair` / `test_fair` = the val / test rows from the mixed sources. There the source can't give the
  answer away. Choose epochs on val_fair and quote test_fair first.

Also writes YOLO files next to pool/: data.yaml (all), data_fair.yaml (fair exams), yolo_train.txt
(healthy photos capped at about 10% so the detector doesn't learn to stay quiet).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

P = next(p for p in Path(__file__).resolve().parents if (p / "datasets").is_dir()) / "datasets" / "oral_datasetv1.2" / "pool"
MIXED = ["dentalai", "zenodo", "kaggle_oral", "external_clean"]
SEED = 42
FRAC = {"normal": (0.70, 0.15, 0.15), "mixed_healthy": (0.50, 0.25, 0.25)}
NAMES = (P / "classes.txt").read_text().split()


def main() -> None:
    d = pd.read_csv(P / "pool.csv")
    m = d.source_file.str.extract(r"anonymous_(\d+)[-_](\d+)[-_](\d+)")
    patient = (m[0] + "-" + m[1] + "-" + m[2]).where(m[0].notna())

    # units = connected groups of (group, patient)
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g, p in zip(d.group, patient):
        a = find(f"g{g}")
        if isinstance(p, str):
            parent[a] = find(f"p{p}")
    d["unit"] = [find(f"g{g}") for g in d.group]

    # one stratum per unit: its most common (source, has_issue)
    d["stratum"] = d.dataset + "|" + d.has_issue.astype(str)
    unit_stratum = d.groupby("unit").stratum.agg(lambda s: s.value_counts().index[0])
    unit_size = d.groupby("unit").size()

    rng = np.random.default_rng(SEED)
    split_of = {}
    for stratum, units in unit_stratum.groupby(unit_stratum):
        src, issue = stratum.split("|")
        f_train, f_val, f_test = FRAC["mixed_healthy" if (src in MIXED and issue == "0") else "normal"]
        u = units.index.to_numpy().copy()
        rng.shuffle(u)
        total, n = unit_size[u].sum(), 0
        for k in u:
            frac = n / total
            split_of[k] = "test" if frac < f_test else "val" if frac < f_test + f_val else "train"
            n += unit_size[k]
    d["split"] = d.unit.map(split_of)

    # sanity: no unit in two splits
    assert d.groupby("unit").split.nunique().max() == 1

    out = d.drop(columns=["unit", "stratum"])
    out.to_csv(P / "pool.csv", index=False)

    # --- YOLO lists ---
    root = P
    tr = out[out.split == "train"]
    healthy = tr[tr.has_issue == 0]
    keep_h = healthy.sample(min(len(healthy), int(0.11 * (tr.has_issue == 1).sum())), random_state=SEED)
    yolo_train = pd.concat([tr[tr.has_issue == 1], keep_h]).image
    lists = {
        "yolo_train.txt": yolo_train,
        "val.txt": out[out.split == "val"].image,
        "test.txt": out[out.split == "test"].image,
        "val_fair.txt": out[(out.split == "val") & out.dataset.isin(MIXED)].image,
        "test_fair.txt": out[(out.split == "test") & out.dataset.isin(MIXED)].image,
    }
    for name, imgs in lists.items():
        (root / name).write_text("\n".join("./" + i for i in sorted(imgs)) + "\n")
    names = "[" + ", ".join(f'"{n}"' for n in NAMES) + "]"
    head = "# paths are relative to this file's folder (no 'path' key on purpose)\n"
    (root / "data.yaml").write_text(head + f"train: yolo_train.txt\nval: val.txt\ntest: test.txt\nnc: {len(NAMES)}\nnames: {names}\n")
    (root / "data_fair.yaml").write_text(head + f"train: yolo_train.txt\nval: val_fair.txt\ntest: test_fair.txt\nnc: {len(NAMES)}\nnames: {names}\n")

    kind = out.has_issue.map({0: "healthy", 1: "problem"})
    t = pd.crosstab([out.dataset], [out.split, kind])
    print(t)
    print(out.groupby("split")[NAMES].sum())
    print({k: len(v) for k, v in lists.items()})
    (P / "split_summary.txt").write_text(f"{t}\n\n{out.groupby('split')[NAMES].sum()}\n\n"
                                         f"{ {k: len(v) for k, v in lists.items()} }\n")


if __name__ == "__main__":
    main()
