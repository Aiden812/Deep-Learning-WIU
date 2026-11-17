# YOLO caries: AI work handover

**Start here.** Everything Claude (AI) did on the YOLO caries detector on 4–6 and 10–13 Nov 2026: what was done, what was found, where the files are, and what to do next. All runs here are **AI feasibility runs, not the submitted model**: reproduce them in your own notebook and log them in your AI Use Log (`docs/ai_use_logs/ang_jun_wei.md`).

## Status at a glance

| Item | State |
|---|---|
| Training environment (`.venv`, GPU) | Done: torch 2.6.0+cu124, ultralytics 8.4.202, RTX 4050 detected |
| Zenodo split (`scripts/Junwei/dedup_and_split.py`) | Done, leak fixed, tests pass. Tightened on 10 Nov (28 photos moved, see step 10) |
| Zenodo-only model | Test mAP50 **0.924**, but **0.087** on external photos (domain shift) |
| Mixed model (Zenodo + oral-diseases caries) | Zenodo test **0.921**, oral-diseases test **0.736**, Kaggle caries flagged **97%**, Kaggle normal flagged **36%** |
| Healthy-photo dataset (`scripts/Junwei/build_mixed.py`) | Trained on 10 Nov (`feasibility_ai_healthy`): healthy clinic photos flagged **2%** (was 16%), Kaggle Normal **34%** (was 41%), Kaggle caries **93%** (was 97%), Zenodo test **0.920**. See step 14 |
| Kaggle Normal in training (`feasibility_ai_kaggle_normal`) | Trained on 11 Nov, **best so far**: Kaggle Normal flagged **16%** (was 38%), Kaggle caries **95%**, Zenodo test **0.920**. See step 16 |
| Daniel's 5-class `oral_dataset` (12–13 Nov) | **Best setup: YOLO11s, 100 epochs** (`feasibility_ai_oral5_s100`): test mAP50 **0.835** / 50-95 **0.450**. Calculus is the weak class (0.674). See steps 17–21 |
| Your own training notebook | **Not started**: use `notebooks/02_train_DRAFT_ai.ipynb` as a reference only |

## Results of all runs

| Run | Training data | Zenodo test mAP50 / 50-95 | oral-diseases test mAP50 / 50-95 | Kaggle caries flagged | Kaggle normal flagged |
|---|---|---|---|---|---|
| `feasibility_ai_leaky` (superseded) | Zenodo, patient-ID split (leaky) | 0.944 / 0.746 | not run | not run | not run |
| `feasibility_ai` | Zenodo, person split | 0.924 / 0.723 | 0.087 / 0.027 | 74% | 44% |
| `feasibility_ai_mixed` | Zenodo + oral-diseases caries | 0.921 / 0.725 | 0.736 / 0.378 | **97%** | **36%** |

All: `yolo11n.pt`, imgsz 640, batch 16, lr0 0.01, up to 100 epochs with patience 20. "Flagged" = at least one caries box at confidence 0.25. Kaggle has one label per photo and unverified "normal" labels, so its rates are indicative, not exact. For the mixed model the oral-diseases test is no longer external (its train split was used); **Kaggle caries vs normal is the only true external check**.

## What was done, in order

1. **Environment.** Created `.venv` at the repo root with the CUDA 12.4 build of PyTorch (the driver supports CUDA 12.5), Ultralytics and `ipykernel`. Set Ultralytics `datasets_dir` to the repo root. Pick the `.venv` kernel in VS Code.
2. **Split script** (`scripts/Junwei/dedup_and_split.py`). Pairs Zenodo labels with images, drops 16 images with boxes outside the image, merges D/d into class 0, splits train/val/test, writes `data/processed/yolo/` and `manifests/yolo_split.csv`. `patient_id()` was written by you; the rest by AI.
3. **The 4,038 unannotated Zenodo images are not healthy.** 1,000 of 1,001 people with both annotated and unannotated photos have caries boxes; annotators labelled ~90% of biting-surface views and ~5–10% of front/side views; checked caries-free photos got an empty label file, these got none. Kept out of training.
4. **Leak found and fixed.** Zenodo gives one person a new, neighbouring ID per session (with/without cheek retractors), photographed minutes apart, e.g. `003-103-297-01` and `003-103-298-00`. 293 such pairs crossed splits. `person_groups()` now links IDs with opposite sessions, numbers ≤ 2 apart and photos ≤ 30 min apart, and the split is by person group: 419 groups (train 293 / val 63 / test 63). A group is not always one person: people photographed back to back get chained into one group, which is safe. 512 IDs have session code 00, so there are about 510 people or more. Leak check after the fix: 0 pairs across splits. Tests: `tests/test_dedup_and_split.py`.
5. **Zenodo-only feasibility run.** Test mAP50 0.924. The leak had inflated it by only ~0.02.
6. **External test.** The same model on oral-diseases (mAP50 0.087) and Kaggle (normal photos flagged 44%). Diagnosis: **domain shift**, not overfitting. The model learned caries in Zenodo-style clinical photos only.
7. **oral-diseases dataset evaluated** (`data/raw/oral-diseases.v2i.yolov11`, Roboflow, CC BY 4.0, 6 classes with per-lesion boxes). ~9,155 distinct photos (Roboflow tripled train). Problems: calculus/gingivitis/hypodontia leak across its own splits (~80% near-copies); calculus and hypodontia photos are 100% Kaggle oral photos (your classifier teammate's data); photos were augmented before labelling (rotations, smeared edges), web/stock quality. Caries photos: no overlap with Kaggle, Zenodo or across its splits (caveat: arbitrary rotations can hide copies).
8. **Mixed run.** Zenodo split + oral-diseases caries (one copy per source photo; class 1 → 0) + 350 background photos from discoloration/ulcer only. Kept Zenodo performance, learned the second style, raised Kaggle caries detection, lowered Kaggle normal false alarms a little. It also flags more Kaggle calculus/discoloration/hypodontia photos (72% / 49% / 46%); some may contain real caries.
9. **Practical lesson:** with 15 GB RAM, training on Windows crashed once (`cuDNN ... HOST_ALLOCATION_FAILED`). Close Chrome/Discord/Spotify and use `workers=2–4`.
10. **Split tightened (10 Nov).** Two rules were added to `dedup_and_split.py`. (a) An ID left alone in its group joins the group of the nearest opposite-session ID with the same prefix, if photographed within 3 hours: it may be a second session the strict rule missed. (b) Anyone photographed while the pilot was still running (up to 2 Oct 14:40) goes to train, because pilot photos have no ID and are all in train. The shuffle is unchanged, so only 17 IDs (28 of 2,211 photos) moved: now 390 groups, train 1,646 / val 263 / test 302 photos. The old split is kept as `results/Junwei/ai_feasibility/feasibility_ai/yolo_split_v2.csv`; the three logged runs used it.
11. **Healthy photos added (10 Nov).** The mixed model had almost never seen a healthy mouth. `scripts/Junwei/build_mixed.py` now adds photos from `mendeley_noncaries` (9,562 clinic photos of teeth without caries, children aged 1–14, CC BY 4.0) with empty labels: 600 train / 120 val / 400 test, the same number from each of its 8 views. That dataset repeats the same mouth in burst shots, so a photo is skipped if its perceptual hash is within 16 bits of one already taken. Train is now 5,190 photos, 19% of them without caries. Ultralytics scanned all three splits: 0 corrupt.
12. **Kaggle halved (10 Nov).** Kaggle is never trained on. Its caries and Normal photos (184 and 247 after dropping look-alikes) are halved into `tune` and `report` in `manifests/yolo_kaggle_split.csv`. Pick the confidence cut-off on `tune`; report once on `report`.
13. **Before-numbers.** `scripts/Junwei/eval_mixed.py` scored the old mixed model (`feasibility_ai_mixed`) so the new run has something to beat: Zenodo test (new split) mAP50 0.919, oral-diseases test 0.736, and photos flagged:

    | Cut-off | Healthy Mendeley test (400) | Kaggle Normal, tune (123) | Kaggle caries, tune (92) |
    |---|---|---|---|
    | 0.25 | 16% | 41% | 97% |
    | 0.4 | 10% | 30% | 95% |
    | 0.5 | 8% | 23% | 89% |
    | 0.6 | 5% | 15% | 79% |

    The old model already leaves most clinic-style healthy photos alone (16%). Its false alarms are mostly on Kaggle's phone-style photos (41%), and Mendeley photos are clinic-style, so the new run may help less on Kaggle than on Mendeley.
14. **Healthy-photo run (10 Nov).** `feasibility_ai_healthy`: same settings as the mixed run, `seed=0`, ran all 100 epochs in 77 minutes (best epoch 73). Zenodo test mAP50 0.920 / 50-95 0.724, oral-diseases test 0.727 / 0.372: box quality is unchanged. Photos flagged, old mixed model → new model:

    | Cut-off | Healthy Mendeley test (400) | Kaggle Normal, tune (123) | Kaggle caries, tune (92) |
    |---|---|---|---|
    | 0.25 | 16% → 2% | 41% → 34% | 97% → 93% |
    | 0.4 | 10% → 1% | 30% → 19% | 95% → 89% |
    | 0.5 | 8% → 1% | 23% → 15% | 89% → 87% |
    | 0.6 | 5% → 1% | 15% → 9% | 79% → 80% |

    False alarms on clinic-style healthy photos are nearly gone. On Kaggle the gain is modest: at the same 89% of caries photos caught, Normal false alarms go from 23% (old, cut-off 0.5) to 19% (new, cut-off 0.4). One Kaggle photo is about 1 point, and this is one seed, so differences of a few points are noise. The `report` half has not been used.
15. **Kaggle Normal added to training (11 Nov).** The brief says the model must be trained and evaluated on the Kaggle Oral Image Dataset, and the remaining false alarms are on Kaggle's phone-style photos. Kaggle has no boxes, so only its Normal photos can be trained on. The 247 distinct Normal photos are now dealt into train 82 / tune 82 / report 83; the 82 train photos are in the mixed train folder with empty labels (train is 5,272 photos). Caries photos stay tune 92 / report 92. The new tune and report parts are subsets of the old halves, so nothing that was held back has been seen. Two things change in how to read the results: the Kaggle Normal number is now an ordinary test, not an outside one, and Kaggle "Normal" labels are not dentist-checked, so a few train photos may show caries. On the smaller tune part the healthy-photo model (step 14) flags Normal 38% / 21% / 16% / 7% at cut-offs 0.25 / 0.4 / 0.5 / 0.6: these are the numbers the new run `feasibility_ai_kaggle_normal` has to beat.
16. **Kaggle Normal run (11 Nov).** `feasibility_ai_kaggle_normal`: same settings and seed, ran all 100 epochs in 81 minutes (best epoch 88). Zenodo test mAP50 0.920 / 50-95 0.727, oral-diseases test 0.726 / 0.370: box quality is unchanged. Photos flagged on the Kaggle tune part, healthy-photo model (step 14) → this model:

    | Cut-off | Healthy Mendeley test (400) | Kaggle Normal, tune (82) | Kaggle caries, tune (92) |
    |---|---|---|---|
    | 0.25 | 2% → 1% | 38% → 16% | 93% → 95% |
    | 0.4 | 1% → 1% | 21% → 12% | 89% → 90% |
    | 0.5 | 1% → 0% | 16% → 9% | 87% → 86% |
    | 0.6 | 1% → 0% | 7% → 5% | 80% → 78% |

    False alarms on Kaggle Normal are more than halved at cut-off 0.25 (31 photos → 13 of 82) and caries detection is unchanged. Part of that drop is the model learning what Kaggle photos look like, because the tune photos come from the same dataset as the 82 it trained on; the caries photos were never trained on, so the caries numbers are still an outside check. One seed. The `report` part has not been used.
17. **Daniel's `oral_dataset` (12 Nov).** 5 classes (Calculus, Caries, Hypodontia, Mouth_Ulcer, Tooth_Discoloration), YOLO labels, ready-made train/val/test, at `data/raw/oral_dataset/` (gitignored). Trained on `yolo_train.txt` (healthy photos capped at 10%) via `data.yaml`; `data_person.yaml` is the same test without photos that have no person ID. The plan is two stages: the team's classifier says normal vs issue, then YOLO shows where. From here on all runs use `workers=3 seed=0`, and `optimizer=auto` picks MuSGD at lr 0.01.
18. **30-epoch baseline and the Qwen check (12 Nov).** `feasibility_ai_oral5_baseline` (YOLO11n, 30 epochs): test mAP50 0.797 / 0.412. Removing all 1,385 train photos with Qwen-drawn boxes (`feasibility_ai_oral5_no_qwen`) gave the same val score (0.774 vs 0.773), so the Qwen boxes neither help nor hurt. `scripts/Junwei/flag_qwen_boxes.py` tried to find the bad Qwen boxes with that model, but marked 74% bad (100% of Zenodo and DentalAI photos, because the model only predicts caries on clinic photos), so the checker is not reliable and nothing was removed.
19. **100 epochs (13 Nov).** `feasibility_ai_oral5_final100` (YOLO11n): ran all 100 epochs, test mAP50 0.827 / 0.440 (+0.030 over 30 epochs). Most of the gain came by epoch 60.
20. **YOLO11s (13 Nov), best so far.** `feasibility_ai_oral5_s100`: same settings with `yolo11s.pt`, about 6.4 min per epoch. Peaked at epoch 68 and stopped early at 88: after about epoch 55 train loss kept falling while val box loss rose (overfitting). Test (person-only test gives the same numbers):

    | Test mAP50 | 11n 30 ep | 11n 100 ep | **11s 100 ep** |
    |---|---|---|---|
    | All | 0.797 | 0.827 | **0.835** |
    | Calculus | 0.596 | 0.646 | **0.674** |
    | Caries | 0.817 | 0.841 | **0.857** |
    | Hypodontia | 0.796 | **0.837** | 0.826 |
    | Mouth_Ulcer | 0.905 | 0.925 | **0.932** |
    | Tooth_Discoloration | 0.870 | 0.886 | 0.886 |
    | mAP50-95 | 0.412 | 0.440 | **0.450** |

    More epochs gave +0.03, a bigger model about +0.01 more, then it overfit: further gains must come from labels, not more training. Hypodontia scores 0.68 on val but 0.83 on test, because test conditions come only from Roboflow, so quote it with that caveat.
21. **Dataset issues to send Daniel.** 14 Zenodo photos wrongly marked Normal; 7 Zenodo IDs in more than one split; test conditions come only from Roboflow (add Zenodo/DentalAI caries to val/test); Qwen drew the non-caries boxes on some Kaggle photos; Mendeley has no person IDs. These make the test more honest; none is expected to raise the score much.
22. **Why Calculus is weak (13 Nov).** The 11s model at cut-off 0.25 finds 63% of the 1,495 test Calculus boxes (IoU 0.5), all from Roboflow. By size: large 85%, medium 68%, small (<1% of the photo) 44%. Of the 552 missed boxes, 385 got no box at all and 167 got Caries (89) or Tooth_Discoloration (78) instead. Looking at 20 photos: brown stains along the gum line are labelled Calculus in some photos, while the model calls the same kind of stain Discoloration or Caries, so the classes overlap in the labels themselves. The test also holds the same photo more than once (rotated or cropped copies). Ideas: ask Daniel how Calculus vs Discoloration was decided; try `imgsz=960` for the small boxes.

## Where everything is

| What | Where |
|---|---|
| Split script, person grouping | `scripts/Junwei/dedup_and_split.py` |
| Mixed dataset builder (Zenodo + oral-diseases + healthy) and Kaggle halves | `scripts/Junwei/build_mixed.py` |
| Scoring script (mAP and photos flagged per cut-off) | `scripts/Junwei/eval_mixed.py` |
| Kaggle train/tune/report parts | `manifests/yolo_kaggle_split.csv` (repo root) |
| Split used by the three logged runs | `results/Junwei/ai_feasibility/feasibility_ai/yolo_split_v2.csv` |
| Its tests | `tests/test_dedup_and_split.py` (repo root) |
| Split manifest (key, patient_id, group, split) | `manifests/yolo_split.csv` (repo root) |
| Draft training notebook (reference only) | `notebooks/02_train_DRAFT_ai.ipynb` |
| Zenodo-only run: README, curves, test plots, leak and 4,038 evidence | `results/Junwei/ai_feasibility/feasibility_ai/` |
| Mixed run: curves, test plots, external-test pictures, its other scripts | `results/Junwei/ai_feasibility/feasibility_ai_mixed/` |
| Experiment log (3 AI rows) | `results/experiments.csv` |
| Weights (gitignored, this laptop only) | `weights/feasibility_ai_*_best.pt`; best 5-class model: `weights/feasibility_ai_oral5_s100_best.pt` |
| 5-class runs: curves, per-class test plots | `runs/detect/feasibility_ai_oral5_*/`, `runs/detect/s100_test_data/`, `runs/detect/final100_test_data/` (repo root) |
| Mixed dataset (gitignored) | `data/processed/yolo_mixed/`, `data/processed/yolo_mixed.yaml`, and `yolo_mixed_od.yaml` (oral-diseases test alone) |

Scripts in `results/Junwei/ai_feasibility/feasibility_ai_mixed/scripts/` were run from a temporary folder and are kept as a record of the old mixed run. `build_mixed.py` moved to `scripts/` and now also adds healthy photos (the old run is the same build without them, on the v2 split). `mixed_train.py` and `external_viz.py` read `external/od_caries.yaml`, which `external_test.py` creates next to itself: to rerun them, run `external_test.py` first in the same folder. The leak and near-duplicate analyses were one-off commands and were not saved as scripts.

## Train the healthy-photo run

This is the command the 10 Nov run used. From the repo root, with Chrome/Discord closed:

```powershell
.venv\Scripts\activate
yolo detect train data=data/processed/yolo_mixed.yaml model=yolo11n.pt imgsz=640 epochs=100 patience=20 workers=2 seed=0 name=feasibility_ai_healthy
python models/yolo_caries/scripts/Junwei/eval_mixed.py --weights <the best.pt path that training prints at the end>
```

Same settings as the old mixed run, so the only change is the data. Compare the output with the before-numbers in step 13, pick a cut-off, then run the last line once more with `--part report` for the final Kaggle numbers.

To rebuild the data from the raw downloads (not needed now):

```powershell
python models/yolo_caries/scripts/Junwei/dedup_and_split.py   # Zenodo split -> data/processed/yolo
python models/yolo_caries/scripts/Junwei/build_mixed.py       # mixed + healthy -> data/processed/yolo_mixed, Kaggle halves
python -m pytest tests/test_dedup_and_split.py
```

## Train the best 5-class setup

From the repo root (about 9.5 hours on the RTX 4050; it stops early when it starts to overfit):

```powershell
yolo detect train data=data/raw/oral_dataset/data.yaml model=yolo11s.pt imgsz=640 epochs=100 patience=20 workers=3 seed=0 name=<your run name>
yolo detect val model=runs/detect/<your run name>/weights/best.pt data=data/raw/oral_dataset/data.yaml split=test
```

Pause with Ctrl+C; resume with `yolo detect train resume model=runs/detect/<your run name>/weights/last.pt`.

## Known limitations

- The old mixed model flags 36% of Kaggle "normal" photos at confidence 0.25: too many false alarms for a screening app. The healthy-photo run lowers it only to 34% (19% at cut-off 0.4).
- Mendeley has no person IDs. Look-alike photos are skipped, but one child can still be in train and test in different views, so its 400 test photos are a rough check. Its "no caries" label is the dataset's own claim, and some photos show stained teeth.
- Only 82 phone-style healthy photos (Kaggle Normal) are in training, next to 600 clinic-style ones. Kaggle caries photos cannot be trained on without drawing boxes on them.
- The mixed model's oral-diseases mAP50-95 (0.378) is low: boxes are found but sit loosely compared with that dataset's boxes.
- Zenodo person groups come from file names (ID, session code, time), not from a published person list. 7 IDs still have no partner within 3 hours. A person who came back on another day under an unrelated ID, or a pilot person photographed again after 2 Oct, cannot be detected.
- Test sets are small (Zenodo: 57 person groups, 302 photos), so treat scores as roughly ±0.02. One seed per run.
- Nothing has been tested on real phone photos from your target users.

## Next steps for you

1. **Commit** the AI work separately from your own edits (guidebook 7.4) and add AI Use Log entries for: the person-level leak fix, the 4,038 finding, the three AI runs, the external test, the oral-diseases evaluation.
2. **Train the healthy-photo run** (section above), then **write your own `notebooks/02_train.ipynb`** and reproduce it. Those are your graded runs.
3. **Tune the confidence cut-off** on the Kaggle `tune` half (`eval_mixed.py` prints 0.25 / 0.4 / 0.5 / 0.6). No retraining needed; it picks the balance between missed caries and false alarms. Report the `report` half once, at the cut-off you picked.
4. **Optimisation runs** for the rubric's top band, one change at a time: `yolo11s.pt`, imgsz 960, adding DentalAI (needs mask → box conversion), stronger blur/colour augmentation.
5. **Update your EDA notebook** with the person-level finding (1,020 IDs are about 510 people, split as 390 person groups) and the 4,038 finding.
6. **Team decision:** oral-diseases has boxes for all 6 conditions, which would allow one detector for everything. Its other classes leak across splits and overlap the classifier's Kaggle data, so they would need re-splitting first.
