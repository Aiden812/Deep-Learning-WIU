# YOLO caries: my to-do

## 1. AI feasibility (now)
- [x] Wait for the `feasibility_ai_healthy` run to finish
- [x] Score it with `eval_mixed.py` and compare with the before-numbers in `AI_HANDOVER.md`
- [x] Wait for the `feasibility_ai_kaggle_normal` run (Kaggle Normal photos added) and compare
- [x] Train on Daniel's 5-class `oral_dataset`: 30-epoch baseline, no-Qwen check, 100 epochs, YOLO11s
- [x] Decide the setup: **YOLO11s, Daniel's dataset as is, 100 epochs, patience 20** (test mAP50 0.835, see `AI_HANDOVER.md` step 20)
- [ ] Send Daniel the fix list (step 21 in `AI_HANDOVER.md`)
- [ ] Ask the lecturer: must my own model train on the Kaggle dataset, or is the team's classifier enough?

## 2. My own work
- [ ] Write my own `notebooks/02_train.ipynb` (type it, do not copy)
- [ ] Run the baseline myself and log it in `results/experiments.csv` under my name
- [ ] Do 2–3 optimisation runs, one change each (epochs 30 → 100, model n → s, then image size or cut-off)
- [ ] Pick the confidence cut-off on the Kaggle `tune` part, report once on `report`
- [ ] Update my EDA notebook (about 510 people, 390 person groups; the 4,038 unlabelled photos)
- [ ] Fill in the AI Use Log: leak fix, healthy photos, feasibility runs
- [ ] Commit the AI work separately from my own

## 3. Defence
- [ ] Be able to explain every line of my notebook
- [ ] Be able to explain: person-level split, mAP50, why healthy photos were added, fine-tuning vs from scratch
- [ ] Practise changing one setting live and rerunning
