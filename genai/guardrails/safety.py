"""Safety checks around the RAG explainer (brief §5 boundaries).

check_input() runs BEFORE search or any LLM call:
  emergency  -> fixed 995 / A&E message (never sent to an LLM)
  diagnosis  -> polite refusal: screening aid only
  medication -> polite refusal: ask a pharmacist / dentist
  injection  -> refusal (attempts to change the assistant's rules)
  personal data (NRIC, phone, email) is removed before anything leaves the server.

check_output() runs on every answer: drops sentences that name medicines or claim a diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY = re.compile(r"""
    (face|facial|cheek|jaw|neck|eye)\W+(\w+\W+){0,3}(swell|swollen|swelling|puff)
  | (swell|swollen|swelling)\W+(\w+\W+){0,3}(face|facial|cheek|jaw|neck|eye)
  | (can'?t|cannot|can\s+not|hard\s+to|difficult(y)?\s+(to\s+)?|trouble)\s*(breath|breathe|breathing|swallow|open\s+my\s+mouth)
  | (bleed|bleeding)\W+(\w+\W+){0,4}(won'?t|not|doesn'?t|does\s+not|can'?t)\s+stop
  | (non.?stop|uncontrolled|heavy|lots\s+of)\s+bleed
  | knock(ed)?\W+(\w+\W+){0,2}(out|loose)|tooth\s+(fell|came)\s+out|(broke|broken|cracked|chipped)\W+(\w+\W+){0,2}(tooth|teeth|jaw)
  | (accident|fell|fall|hit|punched|injur|trauma)\W+(\w+\W+){0,5}(tooth|teeth|mouth|jaw|face)
  | high\s+fever|fever\W+(\w+\W+){0,4}(swell|pain|tooth)
  | unbearable|worst\s+pain|severe\s+pain
""", re.I | re.X)

DIAGNOSIS = re.compile(r"""
    \b(do|does|have|has|got)\s+(i|it|my\s+\w+)\s+(\w+\s+){0,2}(cancer|tumou?r|infection|abscess|disease|hiv|herpes|std)
  | \b(is|could|might)\s+(it|this|that)\s+(\w+\s+){0,2}(cancer|tumou?r|malignant|an?\s+infection|an?\s+abscess|serious|dangerous)
  | \bdiagnos(e|is)\b | \bconfirm\s+(that|if|whether)\b | \bwhat\s+(disease|condition)\s+do\s+i\s+have
""", re.I | re.X)

MEDICATION = re.compile(r"""
    \b(medicines?|medications?|meds|drugs?|pills?|tablets?|antibiotics?|painkillers?|pain\s*killers?|pain\s+relief
      |doses?|dosage|mg|prescri\w*|gels?|creams?|ointments?|lozenges?|numbing)\b
  | \b(paracetamol|panadol|ibuprofen|nurofen|aspirin|amoxicillin|augmentin|metronidazole|clindamycin|penicillin
      |chlorhexidine|corsodyl|difflam|bonjela|oral\s*aid|kenalog|triamcinolone|steroid|benzocaine|lidocaine|orajel)\b
  | \bwhat\s+(can|should)\s+i\s+(take|apply)\b
""", re.I | re.X)

INJECTION = re.compile(r"""
    ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompt)
  | (system|developer)\s+prompt | you\s+are\s+now\b | pretend\s+(to\s+be|you\s+are) | jailbreak | act\s+as\s+(a|an)\s
  | disregard\s+(your|the)\s+(rules|instructions)
""", re.I | re.X)

PII = [
    (re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I), "[NRIC removed]"),
    (re.compile(r"\b[689]\d{3}\s?\d{4}\b"), "[phone removed]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email removed]"),
]

MESSAGES = {
    "emergency": ("This could be a dental emergency. If you have trouble breathing or swallowing, or the swelling is "
                  "spreading, call 995 now. For facial swelling with fever, bleeding that won't stop, or an injured or "
                  "knocked-out tooth, go to an emergency department straight away. NDCS runs 24-hour emergency dental "
                  "care at the SingHealth Outram campus (SGH Emergency after clinic hours)."),
    "diagnosis": ("I can't diagnose conditions. SmileCheck is a screening and education aid, and only a dentist or doctor "
                  "who examines you can say what it is. If something worries you, please book a check-up."),
    "medication": ("I can't recommend medicines or doses. A pharmacist, dentist or doctor can advise you safely, and they "
                   "can check what suits you. I'm happy to explain the condition or what a dental visit involves."),
    "injection": "I can only help with questions about your teeth and this screening result.",
}


@dataclass
class InputCheck:
    category: str | None    # emergency / diagnosis / medication / injection, or None if fine
    message: str | None
    clean_text: str          # the question with personal data removed
    removed_pii: bool


def check_input(text: str) -> InputCheck:
    clean, removed = text, False
    for pat, repl in PII:
        clean, n = pat.subn(repl, clean)
        removed |= bool(n)
    for cat, pat in (("emergency", EMERGENCY), ("injection", INJECTION), ("diagnosis", DIAGNOSIS), ("medication", MEDICATION)):
        if pat.search(clean):
            return InputCheck(cat, MESSAGES[cat], clean, removed)
    return InputCheck(None, None, clean, removed)


# Sentences in an answer that must not reach the patient.
OUT_MEDICINE = re.compile(r"""
    \b(antibiotic\w*|painkiller\w*|dose|dosage|\d+\s?mg|prescri\w*)\b
  | \b(paracetamol|panadol|ibuprofen|nurofen|aspirin|amoxicillin|augmentin|metronidazole|clindamycin|penicillin
      |chlorhexidine|corsodyl|difflam|bonjela|kenalog|triamcinolone|steroid\w*|benzocaine|lidocaine|orajel)\b
""", re.I | re.X)
OUT_DIAGNOSIS = re.compile(r"\byou\s+(have|definitely\s+have|are\s+suffering\s+from)\s+(an?\s+)?(cancer|infection|abscess|caries|gum\s+disease|periodontitis)", re.I)
OUT_OTHER_APP = re.compile(r"oral link|\breport\b|overall risk|\bpriority\b|redirect_links|\b(for\s+)?users\s+(whose|who|with)\b", re.I)


def check_output(text: str) -> tuple[str, list[str]]:
    """Remove unsafe sentences. Returns (clean text, list of reasons for anything removed)."""
    kept, reasons = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if OUT_MEDICINE.search(sent):
            reasons.append("medicine")
        elif OUT_DIAGNOSIS.search(sent):
            reasons.append("diagnosis")
        elif OUT_OTHER_APP.search(sent):
            reasons.append("off-app")
        else:
            kept.append(sent)
    return " ".join(kept), reasons
