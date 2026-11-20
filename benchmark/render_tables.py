"""Save every markdown table in results/benchmark_report.md and results/metric_guide.md as a PNG.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/render_tables.py

Images go to results/tables/, named <file>_<nn>_<heading>.png. Re-run after score.py.
"""
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "tables"
WRAP = 60


def tables(md):
    heading, rows, out = "table", [], []
    for line in md.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip().replace("**", "") for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
            continue
        if rows:
            out.append((heading, rows)); rows = []
        if s.startswith("#"):
            heading = s.lstrip("#").strip().split(";")[0]
    return out


def render(title, rows, path):
    rows = [["\n".join(textwrap.wrap(c, WRAP)) or c for c in r] for r in rows]
    n_lines = [max(c.count("\n") + 1 for c in r) for r in rows]
    widths = [max(max(len(l) for l in r[j].split("\n")) for r in rows) for j in range(len(rows[0]))]
    fig_w = min(26, max(6, sum(widths) * 0.085 + 0.6))
    fig_h = 0.6 + sum(n_lines) * 0.26
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    t = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", colLoc="left", loc="upper left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.auto_set_column_width(list(range(len(rows[0]))))
    for (r, c), cell in t.get_celld().items():
        cell.set_height(n_lines[r] * 0.26 / fig_h)
        cell.set_edgecolor("#c8c8c8")
        if r == 0:
            cell.set_facecolor("#2f4b7c"); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f1f4f9")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    for src in ["benchmark_report.md", "metric_guide.md"]:
        f = RES / src
        if not f.exists():
            continue
        for i, (head, rows) in enumerate(tables(f.read_text(encoding="utf-8")), 1):
            slug = re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_")[:50]
            p = OUT / f"{f.stem}_{i:02d}_{slug}.png"
            render(head, rows, p)
            print(p.name)


if __name__ == "__main__":
    main()
