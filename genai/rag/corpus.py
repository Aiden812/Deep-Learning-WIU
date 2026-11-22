"""Load the knowledge files in genai/rag/corpus/ into small, searchable chunks.

Two formats are in the folder:
  A. "--- CHUNK n ---" blocks with [CHUNK ID] / [TOPIC] / [KEYWORDS] / [TRIGGERS] / CONTENT
     (OralHealthEducation.txt, SupplementConditions.txt)
  B. entries separated by long dash lines: summary, "## Sample queries this entry answers",
     then "# Title" and "## Section" body (the other supplied files)

Each entry is split into chunks of roughly one section. The sample questions / triggers are
kept OUT of the searchable text: they are used as the evaluation set and as suggested questions,
so the retrieval score is not inflated by matching a question to itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# Where each supplied file came from. The supplied files carry no per-entry links.
FILE_SOURCES = {
    "OralHealthEducation.txt": "Project knowledge file (header: HPB Singapore; Singapore Dental Association; last reviewed 2026-04-26)",
    "ProductRecommendation.txt": "Project knowledge file: product guide",
    "ReferralTreatment.txt": "Project knowledge file: treatment explainers",
    "Dental Cost Info.txt": "Project knowledge file: dental costs (figures checked against MOH/CHAS on 2026-11-11)",
    "SupplementConditions.txt": "Written for this project from the official page in each chunk",
}

# Files and entries left out on purpose (see genai/rag/README.md).
SKIP_FILES = {"SampleReportInterpretation.txt"}  # written for another app's report format
SKIP_TITLES = {  # prescription products: the app must not recommend medication
    "Antiseptic mouthwash",
    "Prescription-strength fluoride toothpaste",
}
# Sections that only make sense in the app the files were written for.
SKIP_SECTIONS = {"Curated products"}

# Words that tag a chunk as relevant to one of the screened conditions (title + keywords + opening).
CONDITION_WORDS = {
    "caries": ["cavity", "cavities", "decay", "caries", "filling", "root canal", "fluoride", "sugar", "toothache"],
    "calculus": ["tartar", "calculus", "plaque", "scaling", "cleaning", "interdental", "floss", "bleeding gums", "gum"],
    "tooth_discoloration": ["stain", "discolo", "whitening", "yellow", "smoking"],
    "hypodontia": ["missing tooth", "missing teeth", "hypodontia", "implant", "denture", "bridge"],
    "mouth_ulcer": ["ulcer", "sore"],
}
MAX_CHARS = 1400


@dataclass
class Chunk:
    id: str            # unique chunk id
    entry_id: str      # the entry this chunk belongs to (evaluation target)
    title: str
    section: str
    text: str
    file: str
    keywords: str = ""
    source: str = ""
    url: str = ""
    conditions: list[str] = field(default_factory=list)


@dataclass
class Entry:
    entry_id: str
    title: str
    file: str
    questions: list[str]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def _tag_conditions(title: str, keywords: str, text: str, explicit: str = "") -> list[str]:
    if explicit:
        tags = [c.strip() for c in explicit.split(",")]
        return [c for c in tags if c in CONDITION_WORDS or c == "all"]
    hay = f"{title} {keywords} {text[:400]}".lower()
    return [c for c, words in CONDITION_WORDS.items() if any(w in hay for w in words)]


def _split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on blank lines so no chunk is much longer than `limit`."""
    paras, out, cur = re.split(r"\n\s*\n", text.strip()), [], ""
    for p in paras:
        if cur and len(cur) + len(p) > limit:
            out.append(cur.strip())
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def _field(block: str, name: str) -> str:
    m = re.search(rf"^\[{re.escape(name)}\]\s*(.*)$", block, re.M)
    return m.group(1).strip() if m else ""


def _parse_chunk_format(raw: str, fname: str) -> tuple[list[Chunk], list[Entry]]:
    chunks, entries = [], []
    for block in re.findall(r"--- CHUNK \S+ ---\n(.*?)--- END CHUNK", raw, re.S):
        cid = _field(block, "CHUNK ID")
        title = _field(block, "TOPIC")
        keywords = _field(block, "KEYWORDS")
        src = _field(block, "SOURCE")
        source, url = (src.split("|", 1) + [""])[:2] if src else (FILE_SOURCES.get(fname, fname), "")
        trig = re.search(r"\[TRIGGERS\]\n(.*?)\n\s*\n", block, re.S)
        questions = [q.strip()[2:].strip() for q in trig.group(1).splitlines() if q.strip().startswith("- ")] if trig else []
        body = re.search(r"CONTENT:\n(.*?)(?:\n\[RELATED CHUNKS\]|\Z)", block, re.S)
        text = body.group(1).strip() if body else ""
        conds = _tag_conditions(title, keywords, text, _field(block, "CONDITIONS"))
        for i, part in enumerate(_split_text(text)):
            chunks.append(Chunk(f"{cid}#{i}", cid, title, "", part, fname, keywords, source.strip(), url.strip(), conds))
        entries.append(Entry(cid, title, fname, questions))
    return chunks, entries


def _parse_entry_format(raw: str, fname: str) -> tuple[list[Chunk], list[Entry]]:
    chunks, entries = [], []
    for block in re.split(r"\n-{20,}\s*\n", raw):
        m = re.search(r"^# (.+)$", block, re.M)
        if not m:
            continue
        title = m.group(1).strip()
        if title in SKIP_TITLES:
            continue
        # The treatment explainers were written for another app's report; make the titles neutral.
        title = title.replace("what your report's recommended treatment means", "what the treatment involves")
        eid = _slug(f"{Path(fname).stem}-{title}")
        q = re.search(r"## Sample queries this entry answers\n(.*?)\n\s*\n", block, re.S)
        questions = [re.sub(r'^-\s*"?|"$', "", l.strip()) for l in q.group(1).splitlines() if l.strip().startswith("-")] if q else []
        summary = block[: block.find("---")].strip() if "---" in block[: m.start()] else ""
        body = block[m.end():]
        # One chunk per "## " section (the intro before the first section included).
        sections = re.split(r"^## (.+)$", body, flags=re.M)
        parts = [("", sections[0])] + list(zip(sections[1::2], sections[2::2]))
        n = 0
        for sec, text in parts:
            sec = sec.strip()
            if sec in SKIP_SECTIONS or not text.strip():
                continue
            if not sec and summary and len(text.strip()) < 80:
                text = summary + "\n\n" + text
            for part in _split_text(text):
                conds = _tag_conditions(title, sec, part)
                chunks.append(Chunk(f"{eid}#{n}", eid, title, sec, part, fname, "", FILE_SOURCES.get(fname, fname), "", conds))
                n += 1
        entries.append(Entry(eid, title, fname, questions))
    return chunks, entries


def load(corpus_dir: Path = CORPUS_DIR) -> tuple[list[Chunk], list[Entry]]:
    chunks, entries = [], []
    for path in sorted(corpus_dir.glob("*.txt")):
        if path.name in SKIP_FILES:
            continue
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        parse = _parse_chunk_format if "--- CHUNK " in raw else _parse_entry_format
        c, e = parse(raw, path.name)
        chunks += c
        entries += e
    return chunks, entries


if __name__ == "__main__":
    from collections import Counter

    chunks, entries = load()
    print(f"{len(entries)} entries, {len(chunks)} chunks, {sum(len(e.questions) for e in entries)} sample questions")
    print(Counter(c.file for c in chunks))
    print(Counter(t for c in chunks for t in c.conditions))
    print(max(len(c.text) for c in chunks), "max chunk chars")
