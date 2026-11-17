# External benchmark: dataset audit and plan (2026-11-17)

> **Status 2026-11-17: built.** Code and frozen split in `benchmark/` (see its README), results in
> `benchmark/results/benchmark_report.md`. Changes from the plan below, found while building:
> - The SIFT copy check found more copies than pHash: 78 MetaDent DS1/DS3 photos excluded (web reposts, cropped/edited).
> - MetaDent face portraits (eyes in view, 128 photos) excluded: most DS1 "healthy" photos were face shots, which let
>   models score by "is this a mouth close-up?". After removing them DS1 has 94 problem / 20 healthy photos.
> - AlphaDent: the bench/train split below was frozen (518 bench photos). The train part stays unused until the team
>   decides whether to train on it.
> - Caries boxes got a second, loose score (IoU >= 0.1), because AlphaDent outlines only the lesion while our training
>   boxes often cover the whole tooth.

Goal: compare every team model (Brandon, Aiden, Daniel) on photos **none of them trained on**, then decide what
new data to train on without spoiling that comparison.

## 1. What we have (audited)

Copies were checked with a perceptual hash (64-bit pHash of the photo and of its mirror image, grey letterbox
trimmed) against every training folder on Brandon's laptop: `oral_datasetv1.2/pool` (19,416),
`oral_dataset_new` (37,902, Daniel's data), `oral dataset` Kaggle folders (4,375), Roboflow `v1i` (9,155).
Distance <= 6 = copy. Pairs at 8-12 were checked by eye: all different photos.

| Dataset | What it is | Labels | Copies of our training data | Verdict |
|---|---|---|---|---|
| **AlphaDent** (`raw/alpha-dent`) | 1,320 labelled DSLR clinic photos (5184x3456), retractors, some mirror shots. 293 patients (train 271 / valid 22, no overlap). Test (135) has **no labels** (Kaggle hidden). | Dentist polygons: Abrasion, Filling, Crown, Caries class 1-6 (Black's classes = location, all caries). 910 photos with caries / 410 without; 2.7 caries per photo; lesions are small (~5% of the image side) | **0** | **Use.** Caries boxes + caries photo-level. No calculus/discoloration/ulcer labels, so not usable for problem-vs-healthy. |
| **MetaDent** (`external/MetaDent`) | 60,669 images, but only **2,588 dentist-labelled**. The HF download has no labels; labels are in `MetaDent/labels/metadent_labels.zip` (downloaded 2026-11-17 from the project page). | Per photo: 18 multi-label codes (C1 caries, C6 plaque/calculus, C7 discoloration, C8 missing teeth, C17 ulcer, ...) + polygon contours with free-text descriptions per finding. | **DS2 = the Mendeley teeth set our Kaggle data comes from: 228 of 481 are copies.** DS3 (web): 20 of 1,901. DS1 (hospital): 0. | **Use DS1 + DS3, drop all of DS2.** 1,989 clean photos. |
| **Autooral** (`external/Autooral_dataset`) | 420 ulcer close-ups, 256x256, about 80 patients (up to 57 photos per patient). | Pixel masks + ulcer type (aphthous mild/severe/herpetiform, traumatic, cancerous). **Ulcers only, no negatives.** | **0** | **Use** for ulcer boxes and ulcer recall. Bootstrap by patient. |
| **SMART-OM** (`external/SMART-OM`) | ~4,000 phone/clinic photos by mouth region; Normal / Variation / OPMD / Oral cancer. | Region and lesion polygons for mucosa. | not checked (no overlap expected) | **Normal *mucosa* only** (tongue, cheeks, lips: ~1,890) as ulcer negatives. **Not** healthy teeth: the "Normal" arch photos show stains, caries, calculus and missing teeth. OPMD/cancer is out of scope. |
| **Caries-Spectra** | Single-tooth occlusal crops 224x224: Advanced 800 / Early 800 / None 400. | Folder = class. | 13 copies (web-scraped) | **Skip.** About 3x augmented (800 files ~ 266 unique photos), tooth-crop domain our full-mouth models never see, some "None" crops show stained fissures. |

### Clean MetaDent counts (DS1 + DS3, natural photos, copies removed)

| Our class | MetaDent code | DS1 (hospital) | DS3 (web) | Total |
|---|---|---|---|---|
| Caries | C1 | 33 | 125 | 158 |
| Calculus | C6 (*plaque or calculus*) | 34 | 192 | 226 |
| Tooth discoloration | C7 | 55 | 336 | 391 |
| Hypodontia / missing teeth | C8 (partial edentulism) | 26 | 155 | 181 |
| Mouth ulcer | C17 | 1 | 17 | 18 (too few: use Autooral) |
| Healthy | no codes at all | 61 | 394 | 455 |
| Out-of-scope only (gums, wear, ortho, fillings, crowns) | | 41 | 760 | 801 (left out of the main score) |

Caveats to state in the report:
- C6 merges plaque with calculus, so our Calculus class will look worse than it is.
- C8 is missing teeth in adults, not congenital hypodontia (our Hypodontia class is really "missing teeth" anyway).
- MetaDent's 18 codes were derived from the dentists' free-text notes by an LLM; the contours and notes are the dentists'.
- DS3 healthy photos include stock "smile" shots, while DS3 problem photos are more clinical. That's a style shortcut,
  so **DS1 (hospital, same camera for both) is the fair exam** and is reported first, DS3 second.
- AlphaDent's caries-free photos still often have abrasion/fillings: fine for "caries yes/no", not for "healthy".

### Housekeeping
- No `SOURCE.txt` yet in any folder (link, licence, date). AlphaDent's licence (Kaggle competition rules) and
  SMART-OM's licence are **not checked yet**. MetaDent: CC BY-NC 4.0 (DS1); Autooral: MIT per its repo.
- Only Brandon's weights are on this laptop. Aiden's and Daniel's are needed before their models can be scored.

## 2. Freeze a split first (bench part vs train part)

Current models haven't seen any of these photos, so they could be scored on all of them. But we also want to train
on some of them later. If we freeze a **bench** part now and only ever score on that part, later retrained models
stay comparable with today's. Split by patient (or by near-duplicate cluster for web photos), fixed seed, committed
as `benchmark/manifest.csv` before anyone trains on anything.

| Dataset | Bench (never trained on) | Train part (may be used later) |
|---|---|---|
| AlphaDent | `valid` (22 patients) + ~100 train patients: ~550 photos | the other ~170 patients: ~770 photos |
| MetaDent | all of DS1 (206) + half of DS3 by cluster (~890) | other half of DS3 (~890) |
| Autooral | Val + Test + 1/3 of train patients: ~150 photos | rest: ~270 photos |
| SMART-OM normal mucosa | half the patients (~950) | other half (~950) |

**Open decision:** AlphaDent is the only large dentist-outlined caries set. Splitting it as above gives the caries
detector real training boxes, but retrained models then know AlphaDent's camera style, so their AlphaDent score is
"seen domain". MetaDent caries (DS1 + bench DS3) then stays the truly external caries test. The other option is to keep
all of AlphaDent bench-only (no AlphaDent training).

## 3. Benchmark tracks

| # | Track | Data | Ground truth | Metric |
|---|---|---|---|---|
| 1 | **Problem vs healthy** (the gate) | MetaDent bench, DS1 then DS3 | problem = any of C1/C6/C7/C8/C17; healthy = no codes; out-of-scope-only photos left out (secondary: count them as healthy) | AUROC; balanced accuracy, sensitivity and specificity at each model's **own saved threshold** |
| 2 | **Per condition, photo level** | MetaDent bench (caries, calculus, discoloration, missing teeth); AlphaDent bench (caries) | code present / absent | AUROC per class (one-vs-rest) + sensitivity/specificity at the saved threshold |
| 3 | **Caries boxes** | AlphaDent bench (Caries 1-6 merged; abrasion/filling/crown ignored) | dentist polygons -> boxes | AP50, recall at the model's own conf |
| 4 | **Ulcer** | Autooral bench (positives) and SMART-OM bench normal mucosa (negatives), scored **separately** because the two sets look different (256px crops vs full photos) | masks -> boxes | ulcer recall, box AP50 on Autooral; false-alarm rate on SMART-OM |
| 5 | **Cost** | 100 fixed photos | | CPU ms/photo, model size MB |

Rules: no tuning on the bench (thresholds come from each model's own val set); 95% CIs by bootstrap over patients
(AlphaDent and Autooral patient ids, MetaDent photos), 1,000 resamples; paired bootstrap for "model A beats B";
run the image-size/shape decision-tree cheat check on Track 1 like v1.2 did.

## 4. Implementation

```
benchmark/
  build_manifest.py   raw folders -> dedupe -> bench/train split -> manifest.csv + labels/ (YOLO txt, our 5 class ids)
  dedupe.py           pHash (done) + Aiden's DINOv2 + SIFT matcher (catches crops/rotations pHash misses)
  adapters/           one file per model: load(weights) and predict(PIL image) -> {p_problem, p_class[5], boxes}
                      brandon_cls, brandon_yolo, aiden_gate, aiden_yolo, daniel_cls, daniel_yolo
  run.py              models x manifest -> predictions/<model>.parquet (cached, resumes after a laptop shutdown)
  score.py            tracks 1-5 + bootstrap CIs -> results/benchmark_<date>.md + plots
```

Steps:
1. Add `SOURCE.txt` to each dataset; check AlphaDent and SMART-OM licences.
2. `build_manifest.py`: AlphaDent polygons -> Caries boxes; MetaDent codes -> our classes, contours -> boxes
   (keyword match on the dentist's note, checked by eye on a sample); Autooral masks -> one box per connected part.
   Pre-shrink AlphaDent to 2,048 px on the long side once (only 1.5 GB to read instead of 4.7 GB per run).
3. Second dedupe pass with the SIFT matcher against **every** member's training list (ask Aiden/Daniel for theirs if
   they differ from `oral_dataset_new`), eye-check anything borderline, then freeze and commit the manifest
   (labels and manifest only, photos stay gitignored).
4. Adapters for Brandon's two models first (weights are here) and check them against their own test_fair numbers so
   preprocessing is proven identical, then Aiden's and Daniel's when their weights arrive.
5. `run.py` + `score.py`, one report with fair-exam numbers first.

## 5. More training data (only from the "train" parts above)

| Source | Adds | Notes |
|---|---|---|
| MetaDent DS3 train half (~890) | Real dentist outlines for caries, calculus, discoloration, missing teeth on varied web photos | Contours -> YOLO boxes by keyword; best new data for the weak classes (calculus/discoloration/missing were near 0 on test_fair, mostly Qwen boxes) |
| Autooral train part (~270) | Dentist ulcer masks -> boxes | Many photos per patient: weight or cap per patient |
| SMART-OM normal mucosa train half (~950) | Hard negatives for ulcer (tongue papillae, cheek lines) | Empty label files |
| AlphaDent train part (~770), only if chosen | Dentist caries boxes (~2,500 lesions) | See the open decision in section 2 |
| Unlabelled MetaDent (~58k) | Self-training / pseudo-labels | Low priority; DS3 is web data with mixed licences |
| Not useful | MetaDent DS2 (already in our Kaggle data), Caries-Spectra, SMART-OM arches/OPMD/cancer | |
