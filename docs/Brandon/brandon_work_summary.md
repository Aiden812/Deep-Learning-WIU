# Brandon's work: what was built, why, and what it showed

**IT2226 AI Innovator and Deep Learning Project: Dental Health Awareness and Affordability in Singapore**
Covers 2026-11-03 to 2026-11-17. Git author: `MoonElf0`, branch `brandon`, merged into `dev`.

This document summarises everything I contributed: the SmileCheck SG frontend and backend, the photo quality
check, the cost estimator, the cleaned dataset v1.2, my two models (a "problem or healthy" classifier and a YOLO
detector), the RAG explainer with its safety guardrails, and the external benchmark that scores every team
model. For each piece it gives the reason behind the main decisions and the measured results.

---

## Contents

1. [Summary](#1-summary)
2. [Timeline](#2-timeline)
3. [Frontend: SmileCheck SG prototype](#3-frontend-smilecheck-sg-prototype)
4. [Backend and API](#4-backend-and-api)
5. [Photo quality check](#5-photo-quality-check)
6. [Cost estimator (CHAS / MediSave)](#6-cost-estimator-chas--medisave)
7. [Dataset work: finding the source shortcut and building v1.2](#7-dataset-work-finding-the-source-shortcut-and-building-v12)
8. [Model 1: "problem or healthy" classifier](#8-model-1-problem-or-healthy-classifier)
9. [Model 2: YOLO11n detector](#9-model-2-yolo11n-detector)
10. [RAG explainer and safety guardrails](#10-rag-explainer-and-safety-guardrails)
11. [External benchmark for all team models](#11-external-benchmark-for-all-team-models)
12. [Testing](#12-testing)
13. [Responsible AI, privacy and limitations](#13-responsible-ai-privacy-and-limitations)
14. [Lessons learned](#14-lessons-learned)
15. [Open items](#15-open-items)
16. [Where things are](#16-where-things-are)

---

## 1. Summary

| Area | What I built | Headline result |
|---|---|---|
| Frontend | SmileCheck SG, a patient-facing page: consent, upload/camera, photo angles, quality checklist, results, explanation, cost, clinics map, "Ask about your result" | Covers the brief's Detection, Understanding, Cost and Action gaps |
| Backend | FastAPI app serving the page plus `/api/quality`, `/api/predict`, `/api/estimate`, `/api/ask` | Photos kept in memory only, never written to disk or logged |
| Quality check | Rule-based gate (brightness, blur, size, glare, "shows teeth") | Rejects 100% of very dark and 98% of strongly blurred test copies; ~50 ms on CPU |
| Cost estimator | Out-of-pocket estimate after CHAS and Flexi-MediSave, every fee tied to an MOH/CHAS source | Non-binding range, GST applied the way CHAS specifies |
| Dataset v1.2 | Found that the old dataset could be "passed" by photo size alone; rebuilt it from clean sources with a group split and fair exams | 19,416 photos; a size-only cheat scores near chance on the fair exam |
| Classifier | EfficientNet-B0 "problem or healthy" gate | Fair exam (test_fair) balanced accuracy **0.819**, AUC 0.923 |
| Detector | YOLO11n, 5 conditions | Full test mAP50 **0.712**; fair-exam caries AP50 0.710 |
| RAG | BM25 retrieval over the supplied knowledge files + optional LLM, cited answers | Right entry in top 3 for **89.4%** of 424 questions; safety routing 39/39 |
| Benchmark | Scores every team model on 2,431 external photos none of us trained on | First honest outside test; showed where each model fails |

**The most important finding.** In the team's dataset (v1.1), every Roboflow photo was a problem photo and every
Mendeley photo was healthy, and all old Roboflow photos were exactly 640×640. The rule "exactly 640×640 = problem"
scored **100%** on val and test. Our exams were measuring the photo's source, not the teeth. Everything I did
afterwards (dataset v1.2, fair exams, cheat checks, the external benchmark) came from this.

---

## 2. Timeline

| Date | Work | Commit |
|---|---|---|
| 2026-11-03 | SmileCheck SG frontend prototype; project brief converted to Markdown | `0886b56` |
| 2026-11-04 | FastAPI backend; connected the team's Keras 6-class classifier to the page (replacing the simulated result) | `d758cef` |
| 2026-11-11 | Classifier v1.1 (B1): EfficientNet-B0 "problem or healthy" on Daniel's v1.1 dataset; fee data checked against MOH/CHAS | |
| 2026-11-12 | Photo quality check; cost estimator; RAG explainer, guardrails and evaluations; "Ask about your result" in the page | |
| 2026-11-14 | YOLO v1.1 (Y1); found the 640×640 source shortcut; re-exported Roboflow originals; built and split dataset v1.2; classifier v1.2 (B2) | `478b772` |
| 2026-11-15 to 16 | YOLO v1.2 (Y2) trained overnight (60 epochs, resumable) | `908cad4` |
| 2026-11-17 | External benchmark: dataset audit, copy removal, frozen bench/train split, adapters for every team model, report | `dbfea86` |

All of it is merged into `dev`.

---

## 3. Frontend: SmileCheck SG prototype

**Goal.** One page a member of the public can use: take or upload photos of their mouth, see what the model found
in plain language, understand what it means, see what treatment would cost them, and know where to go next.

**What the page does** (`prototype/Brandon/`)

| Section | What it does |
|---|---|
| Hero | Explains what the tool does and doesn't do (screening, not diagnosis) |
| Screen your teeth | Consent toggle must be on before photos can be added; upload or take a photo with the camera; up to 5 photos, one per angle (front, upper, lower, left, right) |
| Photo checklist | Each photo is checked by `/api/quality` as soon as it is added; the page ticks or crosses each item and explains why a photo was left out |
| Result card | Top condition, a confidence ring ("how closely your photo matches, a dentist must confirm"), likelihood bars for every condition, low-confidence warning, "Delete photos & start again" |
| Understand | Plain-language cause, how it progresses and urgency for the condition (`js/data.js`) |
| Cost | Asks residency, CHAS tier and age group, then calls `/api/estimate` and shows the likely out-of-pocket range with sources |
| Ask about your result | Suggested questions per condition plus free text; answers come from the RAG explainer with citations; emergencies show a red 995 / A&E card |
| Clinics and care | Leaflet map with clinics, distance from the user, filters; home-care product suggestions (no medication) |
| Samples | "Try a sample" chips and `?demo=<condition>` show simulated results (clearly tagged "Sample · simulated result") for demos |

**Design decisions**
- Hand-drawn SVG illustrations of teeth, conditions and photo angles (`js/art.js`) instead of stock photos, so the
  page has no licensing issues and stays light.
- Neumorphic design tokens in `css/styles.css`; works on phone width.
- Consent first, delete button always visible, and wording that never says "you have X": it says "matches" and
  "a dentist must confirm", as the brief requires (probabilities only, no diagnosis).
- Results are shown as probabilities for every condition, not just one label, so the user sees the uncertainty.

---

## 4. Backend and API

`app/backend/main.py` (FastAPI). Run from the repo root:
`python -m uvicorn app.backend.main:app --port 8000`, then open `http://127.0.0.1:8000/Brandon/`.

| Endpoint | What it does |
|---|---|
| `POST /api/quality` | Checks one photo (runs as soon as it is added) |
| `POST /api/predict` | 1–5 photos: quality-checks each, drops the ones that fail, classifies the rest, returns results in upload order |
| `POST /api/estimate` | Cost estimate from condition + residency + CHAS tier + age group |
| `POST /api/ask` | RAG explainer answer; attaches the user's own cost estimate for cost questions |
| `GET /api/ask/suggestions` | Suggested questions for a condition |
| `GET /api/health` | Health check |

I also wrote `models/classifier/src/predict.py` to load the team's Keras classifier (raw 0–255 input, class names
mapped to the IDs in `shared/conditions.yaml`) so the frontend could call a real model from day 2.

**Privacy by design.** Photos are only held in memory for the request; nothing is written to disk. Questions sent to
`/api/ask` are answered and discarded. Upload size is capped at 15 MB per photo. CORS only allows localhost.

---

## 5. Photo quality check

`models/quality_check/src/predict.py`. No trained model: simple measurements with PIL + numpy, ~50 ms per photo on
CPU. Each check returns pass / warn / fail / skip, and only the real cause is reported (a dark photo isn't also
blamed for blur).

| Check | Measured as | Fail | Warn |
|---|---|---|---|
| Big enough | short side, px | < 100 | < 160 |
| Bright enough | 90th-percentile brightness | < 60 | < 95 |
| Not washed out | mean brightness | > 225 | > 205 |
| No flash glare | share of pure-white pixels | – | > 15% |
| In focus | Laplacian variance (second-sharpest block of a 4×4 grid) | < 10 | < 35 |
| Clear detail | brightness standard deviation | < 14 | < 22 |
| Shows teeth and gums | share of gum/lip/tongue or tooth-coloured pixels | < 15% | < 35% |

**How the thresholds were set.** 120 dataset photos plus damaged copies of each:

| Set | pass | warn | fail |
|---|---|---|---|
| Original photos | 95 | 20 | 5 (all under 100 px) |
| Very dark | 0 | 0 | 120 |
| Strong blur | 0 | 2 | 118 |
| Mild blur | 21 | 36 | 63 |
| Colour-swapped ("not a mouth") | 0 | 6 | 114 |

**Why:** the model should never guess from a bad photo. Leaving the photo out, and telling the user why, is safer
than a confident wrong answer.

---

## 6. Cost estimator (CHAS / MediSave)

`prototype/Brandon/backend/costs.py` + `backend/data/fees.json`, adapted from Daniel's prototype.

- Fees cover more procedures, every CHAS tier (Pioneer, Merdeka, Blue, Orange, Green), CHAS claim limits and
  2025 hospital averages. **Every figure has a source and the date it was checked** (MOH / CHAS pages, 2026-11-11).
- GST is applied the way CHAS says: GST-registered clinics add 9% GST *before* the CHAS subsidy. The range runs
  from the cheapest case (lowest fee, no GST) to the dearest (highest fee + GST).
- Alternative treatments (e.g. denture *or* implant) are costed separately.
- Only three self-selected answers are used (residency, tier, age group); nothing is stored.
- Always labelled as a non-binding estimate, never a quote.

---

## 7. Dataset work: finding the source shortcut and building v1.2

### 7.1 What was wrong with v1.1
While evaluating my first models I checked *why* the scores were so high (classifier 0.95, YOLO problem-vs-healthy
0.95):
- Roboflow photos were 100% problem; Mendeley photos 100% healthy.
- In val and test, every problem photo was Roboflow and no healthy photo was.
- The old Roboflow export was stretched to exactly 640×640, ~46% had black corners, with 2–3 augmented copies
  of each photo.
- **The rule "exactly 640×640 = problem" scored 100% on val and test.**

So I created a **fair exam** (`test_fair`): held-out zenodo / DentalAI patients that have *both* healthy and problem
photos from the same source. On it my v1.1 classifier dropped from 0.95 to 0.80.

### 7.2 Roboflow re-export
I forked the Roboflow project and made a new version with **Auto-Orient only, no resize, no augmentation**
(9,155 originals, 0 at 640×640). I verified every one of the old 8,958 unique photos is in it, and that old copy 1
is the new photo stretched to 640×640 (median pixel difference 1.5/255). Side-by-side: `datasets/roboflow_old_vs_new.png`.

### 7.3 Dataset v1.2 (built 2026-11-14)
Scripts in `models/classifier/notebooks/Brandon/dataset_v12/`:
`build_v1_2.py` (pool), `find_rf_copies.py --apply` (copy removal), `split_v1_2.py` (split + YOLO files).

| Step | Result |
|---|---|
| Old Roboflow (stretched, 3× augmented) | all 23,277 rows dropped |
| Clean Roboflow originals | 9,155 in; 197 gingivitis-only removed → 8,958 |
| Near-identical copies across sources | 53 removed |
| Mirrored / recoloured / rotated copies of Roboflow photos (checked by eye) | 51 removed |
| EXIF-rotated photos with uncertain boxes | 4 removed |
| **Final** | **19,416 photos: 13,592 problem / 5,824 healthy** |

- **5 classes** (same IDs as `shared/conditions.yaml`): Calculus, Caries, Hypodontia, Mouth_Ulcer,
  Tooth_Discoloration. Gingivitis was dropped because it isn't one of the app's conditions.
- **Same processing for every source:** longest side 640, padded with grey (never stretched), JPEG q90, no EXIF.
  Boxes moved to match and checked visually (`box_check.png`).
- **Group/patient split:** copies of a photo and every photo of one patient stay together; each source's healthy
  and problem photos are split separately.
- **Fair exams:** `val_fair.txt` (1,297) and `test_fair.txt` (1,022). Epochs are chosen on val_fair and test_fair is
  reported first.
- **Cheat check:** a decision tree that sees only the photo's size and shape. It still does well on the normal
  exams but is close to a coin flip on the fair exams, which shows the fair exams measure the teeth.

---

## 8. Model 1: "problem or healthy" classifier

Stage 1 of the pipeline: is there any of the five conditions in this photo?

| | B1 (v1.1 data) | **B2 (v1.2 data)** |
|---|---|---|
| Notebook | `01_issue_classifier.ipynb` | `03_issue_classifier_v12.ipynb` |
| Backbone | EfficientNet-B0 (ImageNet), 224 px | same |
| Epochs (best) | 12 (10) | 8 (4), chosen on val_fair |
| **Fair exam: balanced accuracy** | 0.795 | **0.819** |
| Fair exam: problem recall / healthy recall | 0.725 / 0.865 | 0.939 / 0.699 |
| Fair exam: AUC | 0.876 | **0.923** |
| Normal test: balanced accuracy | 0.949 (inflated by the shortcut) | 0.930 |
| Weights | `issue_classifier.pt` | `issue_classifier_v12.pt` (16 MB) |

**Decisions**
- Binary first: one "problem?" score is easier to make reliable than 5 labels, and it decides whether to recommend a
  check-up.
- The v1.2 notebook cuts the grey padding back off before training (it knows each photo's original size), so the
  model can't learn from padding shape.
- Temperature scaling for calibrated probabilities (T = 1.77), since the page shows probabilities.
- The two fair exams are different test sets, so B1 vs B2 isn't a perfectly like-for-like comparison.

**Remaining shortcut.** Healthy recall on Mendeley is 0.99 but on DentalAI only 0.70, so the model still partly
recognises the Mendeley photo style as "healthy".

---

## 9. Model 2: YOLO11n detector

Stage 2: draws boxes and names the condition. Trained on a GTX 1050 Ti (4 GB); the notebook resumes from `last.pt`
because the laptop overheats on long runs.

| | Y1 (v1.1 data) | **Y2 (v1.2 data)** |
|---|---|---|
| Notebook | `02_yolo_detector.ipynb` | `04_yolo_detector_v12.ipynb` |
| Model | YOLO11n, 640 px | YOLO11n, 640 px |
| Epochs (best) | 30 (26) | 60 (59), patience 20 |
| Full test mAP50 | 0.685 | **0.712** |
| Fair exam: Caries AP50 (human boxes) | 0.75 | 0.710 |
| Fair exam: image-level balanced acc / AUC | 0.88 (problem vs healthy) | 0.809 / 0.904 |
| Weights | `yolo_detector.pt` (5.5 MB) | `yolo_detector_v12.pt` (5.5 MB) |

**Honest caveats**
- On the fair exam the other four classes score near 0, but those boxes were drawn by Qwen (an AI), not people, so
  that number isn't meaningful. Caries is the only class with human boxes there.
- Kaggle problem recall is only 0.43.
- Figures (PR curves, confusion matrices, training curves, example predictions) are in
  `models/yolo_caries/results/Brandon/figures/`.

---

## 10. RAG explainer and safety guardrails

**Goal.** Close the brief's *Understanding* gap: let the user ask follow-up questions about their result and get an
answer that only uses trusted sources and shows where each point comes from.

```
question ─► guardrails.check_input ─► emergency?            → fixed 995 / A&E message (no search, no LLM)
                                   └► diagnosis / medicine / prompt injection? → polite refusal
         ─► search.py (BM25, boosted to the screened condition) ─► weak or off-topic? → "I don't know"
         ─► answer: LLM told to use only the passages and cite them   (if .env has an LLM)
                    or the best-matching sentences, quoted with citations (no LLM)
         ─► guardrails.check_output (drops medicine names, diagnoses, other-app wording) ─► page
```

| File | What it does |
|---|---|
| `genai/rag/corpus.py` | Splits the knowledge files into 306 chunks (82 entries); sample questions are kept out of the index |
| `genai/rag/search.py` | BM25 keyword search, pure Python, no GPU |
| `genai/rag/answer.py` | The pipeline above + suggested questions per condition |
| `genai/rag/llm.py` | Optional LLM (Anthropic or any OpenAI-compatible server via `.env`) |
| `genai/guardrails/safety.py` | Emergency detection, refusals, PII removal (NRIC, phone, email), output filter |
| `genai/rag/eval/` | Retrieval and safety evaluations |

**Corpus decisions**
- Used the four supplied knowledge files (copied, originals untouched).
- **Left out** `SampleReportInterpretation.txt` (written for another product's report format) and two prescription
  products in `ProductRecommendation.txt` (the brief rules out recommending medication).
- **Added** `SupplementConditions.txt`: the supplied files had nothing on hypodontia and little on ulcers,
  discolouration or emergencies, so I wrote 9 short chunks paraphrased from NDCS, SGH and SingHealth pages.
- With an LLM, an answer is only shown if every citation points at a real passage; otherwise it falls back to quoting.

**Results (2026-11-12)**

| Evaluation | Result |
|---|---|
| Retrieval: right entry in top 1 / 3 / 5 (424 questions) | 60.4% / **89.4%** / 95.3% (MRR 0.754) |
| Safety and routing (39 cases: emergency, diagnosis, medicine, prompt injection, off-topic, normal) | **39/39** |

The 39 safety cases were used while writing the rules, so they are a development set; a fresh held-out set is still
needed for an honest score.

---

## 11. External benchmark for all team models

**Goal.** Compare every team model (Brandon, Aiden, Daniel) on photos **none of us trained on**, so we stop
grading ourselves on our own data. Plan and audit: `docs/benchmark_plan.md`. Code: `benchmark/`.

### 11.1 Dataset audit
| Dataset | Use | Why |
|---|---|---|
| AlphaDent | Caries boxes + caries yes/no | Dentist outlines, clinic photos, 0 copies of our data |
| MetaDent DS1 (hospital) + DS3 (web) | Problem vs healthy, per condition | Dentist labels. **DS2 dropped**: it is the Mendeley set our Kaggle data came from (228 copies) |
| Autooral | Ulcer recall and ulcer boxes | Dentist masks |
| SMART-OM normal mucosa | Ulcer false alarms | Normal tongue/cheek/lip; its arch photos were *not* used as "healthy teeth" because they show stains and caries |
| Caries-Spectra | Skipped | Augmented single-tooth crops, a domain our models never see |

### 11.2 Making it fair
- **Copy removal:** pHash (distance ≤ 6, incl. mirrored) and SIFT (≥ 25 matching keypoints) against every team
  training folder (v1.2, Daniel's 37,902-photo set, Kaggle, Roboflow).
- Removed 128 MetaDent face portraits: most DS1 "healthy" photos were face shots, which let a model score by
  "is this a close-up?".
- **Frozen bench/train split by patient** (`manifest.csv`): bench 2,431 / train 2,723 / excluded 766.
  Rule: never train on a bench row, so later models stay comparable.
- **DS1 (same hospital camera for both groups) is the fair exam and is reported first.**
- A loose caries box score (IoU ≥ 0.1) as well as the strict one, because AlphaDent outlines only the lesion while
  our boxes cover the whole tooth.
- 95% bootstrap confidence intervals over patients; each model scored at its own saved threshold.
- **Adapters** for every team model format (Brandon's/Daniel's torchvision checkpoints, Aiden's gate, any
  ultralytics YOLO, the Keras baseline). `sanity_check.py` confirms the adapters reproduce the notebooks' numbers.

### 11.3 First results (my models + team Keras baseline)
| Test | Best result | What it shows |
|---|---|---|
| MetaDent DS1 hospital, problem vs healthy (AUROC) | YOLO v12 **0.833**; my classifiers 0.54–0.58 | Classifiers are near chance on real clinic photos; YOLO generalises better |
| MetaDent DS3 web (AUROC) | classifier v11 0.807, YOLO v12 0.804 | Reasonable on web photos |
| AlphaDent caries boxes | AP50 0.02, loose AP10 0.11 | Lesion-sized dentist boxes vs our whole-tooth boxes; caries mostly missed |
| Autooral ulcers caught / SMART-OM false alarms | YOLO v12 **72% / 1%** | Good ulcer detector with very few false alarms |
| CPU speed | classifier ~60 ms, YOLO ~120 ms per photo | Both fast enough for the web app |

Aiden's and Daniel's models are wired in but not scored yet (their weights aren't on my laptop). Full tables:
`benchmark/results/benchmark_report.md`.

### 11.4 Retraining on the benchmark's train part: v1.3 (2026-11-19)
The benchmark showed three weaknesses. The classifiers had learned dataset style rather than disease. Our caries boxes cover the whole tooth while dentists outline only the lesion. And the confidence cut-offs were set too high on photos the models had never seen. Three steps fixed these, run overnight by `datasets/v13_work/run_overnight.sh`:

1. **Re-picked thresholds** (`benchmark/calibrate.py`): every model's cut-off was re-chosen for best balanced accuracy on the MetaDent *train* part. These rows appear as `_cal` in the report. YOLO v12 on DS1 went from 66% to 79% balanced accuracy with no retraining.
2. **YOLO v1.3** (`models/yolo_caries/scripts/Brandon/{build,train}_yolo_v13.py`): fine-tuned from v1.2 for 10 epochs at 960 px.
   - Added data: the AlphaDent lesion boxes, MetaDent DS3 contours (turned into boxes only where the dentist code agrees), Autooral ulcer boxes, and SMART-OM normal mucosa as negatives.
   - Dropped data: the 3,551 Zenodo/Roboflow whole-tooth caries photos.
   - The confidence cut-off (0.038) was picked on a 20% held-out MetaDent slice.
3. **Classifier v1.3** (`models/classifier/scripts/Brandon/train_issue_v13.py`): EfficientNet-B0 trained from ImageNet weights.
   - Training batches are balanced across sources, so no dataset can stand in for the answer.
   - AlphaDent is left out because its no-caries photos are not confirmed healthy.

| Test (bench part, never trained on) | v1.2 | v1.3 |
|---|---|---|
| MetaDent DS1 hospital, YOLO AUROC / balanced acc | 0.833 / 66% | **0.937 / 88%** |
| MetaDent DS1 hospital, classifier AUROC | 0.538 | **0.882** |
| MetaDent DS3 web, YOLO / classifier AUROC | 0.804 / 0.771 | 0.795 / 0.819 |
| AlphaDent caries boxes AP50 / lesions found | 0.024 / 7% | **0.310 / 72%** |
| Autooral ulcers caught / ulcer box AP50 | 72% / 0.12 | **96% / 0.67** |
| Classifier v1.2 test_fair balanced acc | 0.819 | 0.860 |

Cautions:
- DS1 has only 20 healthy photos, so its confidence intervals are wide.
- AlphaDent, Autooral and SMART-OM bench photos come from different patients but the same clinics as the train part, so those gains are partly in-distribution. DS1 is the fairest test.
- YOLO v1.3 scores lower on the old v1.2 test_fair_human caries boxes (AP50 0.42 vs 0.71). This is expected: those boxes are the whole-tooth style we stopped training on.

---

## 12. Testing

Logged in `docs/test_log.md`:

| Date | Component | Test | Result |
|---|---|---|---|
| 2026-11-12 | RAG retrieval | 424 un-indexed sample questions | Hit@3 89.4% – Pass |
| 2026-11-12 | RAG safety | 39 routing cases | 39/39 – Pass (development set) |
| 2026-11-12 | RAG UI | Ask chip, typed cost question, emergency question in the page | Cited answer, cost pill, red 995 card – Pass |

Plus: quality-check threshold tests (section 5), cheat checks in every classifier notebook (section 7.3), benchmark
`sanity_check.py` (section 11.2).

---

## 13. Responsible AI, privacy and limitations

- **No diagnosis, no medication:** wording says "matches" and "a dentist must confirm"; the guardrails refuse
  diagnosis and medicine questions and filter medicine names out of answers.
- **Emergencies:** swelling, breathing difficulty, uncontrolled bleeding, knocked-out teeth and severe pain go
  straight to a fixed 995 / A&E message, with no search or LLM in between.
- **PDPA:** consent before upload; photos in memory only; questions not logged; NRIC/phone/email stripped from
  questions; delete button always visible.
- **Fairness / honesty:** found and documented the source shortcut; fair exams and external tests reported first;
  the remaining Mendeley shortcut and the weak classes are stated, not hidden.
- **Limits:** the quality check's "shows teeth" test is colour-based (a pink object can pass); thresholds were
  tuned on dataset photos, not real phone photos; most detector classes have no human-labelled fair boxes; cost
  data must be re-checked when MOH updates its benchmarks; clinic entries on the map are samples.

---

## 14. Lessons learned

1. **High scores need a "why".** 0.95 looked great until a one-line size rule matched it. Always run a cheat check.
2. **Test on photos from somewhere else.** Same-source tests flattered every model; the external benchmark showed
   the classifiers fall to near chance on hospital photos.
3. **Know what your labels are.** AlphaDent's lesion boxes vs our whole-tooth boxes changed AP50 from "bad" to
   "different definition".
4. **Check exports, not just data.** The Roboflow "dataset" was a stretched, augmented version; the originals had
   to be re-exported.
5. **Safety rules go before the model.** Emergencies and refusals are handled by fixed rules, so an LLM never gets
   the chance to get them wrong.

---

## 15. Open items

- Score aiden_yolo11n once its weights are copied over (`benchmark/adapters/__init__.py`).
- Switch the app to the v1.3 classifier and YOLO; measure YOLO v1.3's CPU speed at 960 px.
- Two-stage pipeline (classifier → YOLO) in the app; currently `/api/predict` uses the team Keras classifier.
- A held-out safety test set and an LLM-mode evaluation for the RAG explainer.
- 50–100 consented real phone photos as a real-world test; human check of ~200 Qwen labels.
- Replace sample clinic entries with live CHAS clinic data.

---

## 16. Where things are

| What | Path |
|---|---|
| Frontend | `prototype/Brandon/` (`index.html`, `css/`, `js/app.js`, `js/data.js`, `js/art.js`) |
| Backend | `app/backend/main.py`, `app/requirements.txt` |
| Cost estimator | `prototype/Brandon/backend/costs.py`, `prototype/Brandon/backend/data/fees.json` |
| Quality check | `models/quality_check/` |
| Keras classifier loader | `models/classifier/src/predict.py` |
| Dataset v1.2 scripts | `models/classifier/notebooks/Brandon/dataset_v12/` |
| Classifier notebooks | `models/classifier/notebooks/Brandon/01_issue_classifier.ipynb`, `03_issue_classifier_v12.ipynb` |
| Classifier results / weights | `models/classifier/results/Brandon/`, `models/classifier/weights/Brandon/` |
| YOLO notebooks | `models/yolo_caries/notebooks/Brandon/02_yolo_detector.ipynb`, `04_yolo_detector_v12.ipynb` |
| YOLO results / weights | `models/yolo_caries/results/Brandon/`, `models/yolo_caries/weights/Brandon/` |
| RAG and guardrails | `genai/rag/`, `genai/guardrails/safety.py` |
| Benchmark | `benchmark/` (README, manifest, scripts, `results/benchmark_report.md`), `docs/benchmark_plan.md` |
| Test log | `docs/test_log.md` |

Work was done with Claude (Anthropic) as a coding assistant; each commit records this with a `Co-Authored-By` line.
