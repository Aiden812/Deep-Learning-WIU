# RAG explainer: "Ask about your result"

Answers a patient's follow-up questions about their screening result, using only the knowledge files in
`corpus/`, and shows where each point comes from. It closes the brief's Understanding gap.

```
question ─► guardrails/safety.check_input ─► emergency? → fixed 995 / A&E message (no search, no LLM)
                                           └► diagnosis / medicine / prompt injection? → polite refusal
         ─► search.py (BM25, boosted to the screened condition) ─► weak or off-topic? → "I don't know"
         ─► answer: LLM told to use only the passages and cite them   (if .env has an LLM)
                    or the best-matching sentences, quoted with citations (no LLM)
         ─► guardrails/safety.check_output (drops medicine names, diagnoses, other-app wording) ─► page
```

The question is never stored or logged. NRIC numbers, phone numbers and emails are removed before anything
else happens.

## Files

| File | What it does |
|---|---|
| `corpus/*.txt` | Knowledge files. Four are copies of the supplied files in `datasets/` (the originals are untouched). `SupplementConditions.txt` was written for this project. |
| `corpus.py` | Splits the files into 306 chunks (82 entries). Sample questions are kept out of the searchable text. |
| `search.py` | BM25 keyword search. Pure Python, no GPU or extra packages. |
| `answer.py` | The pipeline above, plus suggested questions per condition. |
| `llm.py` | Optional LLM call (Anthropic or any OpenAI-compatible server, set in `.env`). |
| `eval/eval_retrieval.py` | Retrieval score on the files' own 424 sample questions. |
| `eval/eval_safety.py` | 39 routing and safety cases. |

API (in `app/backend/main.py`): `POST /api/ask {question, condition, residency, tier, age}` and
`GET /api/ask/suggestions?condition=`.

## What was left out, and why

- `SampleReportInterpretation.txt` was written for another product's report ("OVERALL RISK: HIGH", tooth
  numbers, plaque levels), which SmileCheck never shows. Using it would confuse patients.
- "Antiseptic mouthwash" and "Prescription-strength fluoride toothpaste" in `ProductRecommendation.txt` are
  prescription products, and the brief rules out recommending medication.
- "Curated products" sections refer to the other app's product list.

## What was added

The supplied files had nothing on hypodontia and almost nothing on mouth ulcers, discolouration or dental
emergencies. `SupplementConditions.txt` adds 9 short chunks, paraphrased from these official pages
(checked 2026-11-12):

- NDCS: [Discoloured Teeth](https://www.ndcs.com.sg/symptoms-treatments/discoloured-teeth)
- SGH: [Mouth Ulcers](https://www.sgh.com.sg/symptoms-treatments/mouth-ulcers)
- NDCS: [Hypodontia article](https://www.ndcs.com.sg/news/patient-care/tried-counting-how-many-teeth-your-children-have-they-might-naturally-be-missing-a-few) (26 Jun 2024)
- NDCS: [Missing Teeth and Tooth Replacement](https://www.ndcs.com.sg/symptoms-treatments/missing-teeth-tooth-replacement-dental-implants-bridges-dentures)
- NDCS: [Emergency Dental Services](https://www.ndcs.com.sg/clinic-visit/emergency-dental-services)
- SingHealth GPFirst: [Gum or Tooth Pain](https://www.singhealth.com.sg/community-care/find-resources-in-your-community/what-is-gpfirst/conditions-your-gp-can-treat/gum-or-tooth-pain)

## Results (2026-11-12)

- Retrieval: right entry in the top 3 for **89.4%** of 424 questions (top 5: 95.3%, top 1: 60.4%).
  Top-1 is lower because several entries overlap (for example a treatment explainer and its cost entry).
- Safety and routing: **39/39**. These cases were used while writing the rules, so they are a development
  set. A fresh held-out set is still needed for an honest score.

Re-run from the repo root:

```bash
python -m genai.rag.eval.eval_retrieval
python -m genai.rag.eval.eval_safety
```

## Known limits

- Without an LLM, answers are quoted sentences. They are accurate to the sources but can read choppily,
  especially for cost questions, where the figures sit in tables. The cost pill under the answer shows the
  person's own estimate from the cost section.
- With an LLM, answers are only shown if every citation points at a real passage; otherwise it falls back
  to quoting. LLM answers have not been evaluated yet (no provider configured).
- The supplied files have no per-entry source links. Their citations name the file only.
