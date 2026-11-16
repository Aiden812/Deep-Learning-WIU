"""BM25 keyword search over the corpus chunks. Pure Python: no GPU, no extra packages.

Title and keywords count double. When the screening result is known, chunks tagged with that
condition get a small boost, so "will it hurt?" after a caries result prefers filling content.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Chunk, load

STOP = set("""a an and are as at be been but by can could do does did for from had has have how i if in into is it
its me my of on or our should so than that the their them then there these they this to too us was we were what when
where which who why will with would you your yours i'm it's i've don't can't""".split())

# Everyday words patients use -> words the corpus uses.
SYNONYMS = {
    "cavity": "decay caries", "cavities": "decay caries", "hole": "cavity decay", "rotten": "decay",
    "tartar": "calculus plaque", "calculus": "tartar", "yellow": "stain discoloured", "stained": "stain",
    "discolouration": "discoloured stain", "discoloration": "discoloured stain", "whiten": "whitening",
    "gap": "missing", "hypodontia": "missing", "sore": "ulcer", "ulcers": "ulcer", "price": "cost fee",
    "expensive": "cost fee", "cheap": "cost fee", "pay": "cost fee subsidy", "hurt": "pain", "painful": "pain",
    "toothpaste": "toothpaste fluoride", "cleaning": "scaling polishing", "clean": "cleaning",
    "cleaned": "cleaning scaling polishing", "brush": "brushing technique", "checkup": "check-up recall",
    "grey": "discoloured colour darker", "gray": "discoloured colour darker", "dark": "discoloured colour",
    "colour": "discoloured stain", "color": "discoloured stain",
}


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
    return [_stem(w) for w in words if w not in STOP and len(w) > 1]


def expand(query: str) -> list[str]:
    words = re.findall(r"[a-z]+", query.lower())
    extra = " ".join(SYNONYMS[w] for w in words if w in SYNONYMS)
    return tokens(query + " " + extra)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.4, b: float = 0.7):
        self.chunks = chunks if chunks is not None else load()[0]
        self.k1, self.b = k1, b
        self.docs = []
        for c in self.chunks:
            head = f"{c.title} {c.section} {c.keywords}"
            self.docs.append(Counter(tokens(head) * 2 + tokens(c.text)))
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5, condition: str | None = None, per_entry: int = 2) -> list[Hit]:
        q = expand(query)
        if not q:
            return []
        hits = []
        for c, d, L in zip(self.chunks, self.docs, self.lens):
            s = 0.0
            for t in set(q):
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            if s and condition and (condition in c.conditions or "all" in c.conditions):
                s *= 1.15
            if s:
                hits.append(Hit(c, s))
        hits.sort(key=lambda h: -h.score)
        # Keep at most `per_entry` chunks of any one entry so the answer draws on several entries.
        out, seen = [], Counter()
        for h in hits:
            if seen[h.chunk.entry_id] < per_entry:
                out.append(h)
                seen[h.chunk.entry_id] += 1
            if len(out) == k:
                break
        return out
