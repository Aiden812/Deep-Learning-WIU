"""Retrieval evaluation: does search find the right entry for each entry's own sample questions?

The sample questions ("Sample queries this entry answers" / [TRIGGERS]) are never indexed, so this
measures real matching, not a question finding itself. Writes genai/rag/eval/retrieval_results.md.

Run from the repo root:  python -m genai.rag.eval.eval_retrieval
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

from genai.rag.corpus import load
from genai.rag.search import Index

OUT = Path(__file__).resolve().parent / "retrieval_results.md"
KS = (1, 3, 5)


def main() -> None:
    chunks, entries = load()
    index = Index(chunks)
    totals = defaultdict(int)
    by_file = defaultdict(lambda: defaultdict(int))
    misses = []
    rr = 0.0
    for e in entries:
        for q in e.questions:
            ranked = []
            for h in index.search(q, k=10, per_entry=1):
                ranked.append(h.chunk.entry_id)
            rank = ranked.index(e.entry_id) + 1 if e.entry_id in ranked else None
            totals["n"] += 1
            by_file[e.file]["n"] += 1
            rr += 1 / rank if rank else 0
            for k in KS:
                if rank and rank <= k:
                    totals[k] += 1
                    by_file[e.file][k] += 1
            if not rank or rank > 3:
                top = index.search(q, k=1)
                misses.append((q, e.title, top[0].chunk.title if top else "-", rank))

    n = totals["n"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Run {date.today()} · {len(entries)} entries · {len(chunks)} chunks · {n} questions "
        "(each entry's own sample questions, which are not indexed).",
        "",
        "A hit = the entry the question was written for is among the top k distinct entries returned.",
        "",
        "| Set | Questions | Hit@1 | Hit@3 | Hit@5 |",
        "|---|---:|---:|---:|---:|",
        f"| **All** | {n} | {totals[1]/n:.1%} | {totals[3]/n:.1%} | {totals[5]/n:.1%} |",
    ]
    for f, t in sorted(by_file.items()):
        lines.append(f"| {f} | {t['n']} | {t[1]/t['n']:.1%} | {t[3]/t['n']:.1%} | {t[5]/t['n']:.1%} |")
    lines += ["", f"Mean reciprocal rank (top 10): {rr/n:.3f}", "",
              f"## Questions not in the top 3 ({len(misses)})", "",
              "| Question | Expected entry | Top result instead | Rank |", "|---|---|---|---:|"]
    for q, exp, got, rank in misses:
        lines.append(f"| {q} | {exp} | {got} | {rank or '>10'} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12 + len(by_file)]))
    print(f"... {len(misses)} misses listed in {OUT}")


if __name__ == "__main__":
    main()
