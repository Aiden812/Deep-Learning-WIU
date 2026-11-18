# Supervised Model: YOLO Detector (Jun Wei)

**Closes the brief's detection gap:** draws a box around each condition, so one photo can show several problems.
Kaggle gives one label per photo and no boxes, so a detector needs extra boxed data.

| | **Dataset checks** (Zenodo caries photos) | **Training** (team dataset, 5 conditions) |
|---|---|---|
| **Data** | 6,265 clinic photos, 2,245 with caries boxes | ~32k photos (26k train / 2.9k val / 2.9k test): Kaggle plus boxed public sets |
| **What I did** | Checked labels, boxes, duplicates and patients per split | Fine-tuned YOLO11n from pretrained weights, 100 epochs, at 640 px and 960 px with the same settings |
| **Found / result** | 4,038 photos unlabelled (not "healthy") · 16 photos with boxes off the image · authors' split puts 280 of 1,020 patients in more than one split | **Test mAP50 0.814**, mAP50-95 0.425, precision 0.78, recall 0.76 (640 px; validation 0.808) |

**Key finding: same person on both sides of the split.** Fixed by splitting by person, not by photo: 1,020 patient IDs → 390 person groups, and no known pair crosses train and test.

**Optimisation: image size 640 → 960 px** (same settings, 100 epochs each): test mAP50 0.814 → 0.811 (−0.003); validation was +0.006. 960 px learns faster (+0.022 at epoch 25), but 640 px catches up. The EDA explains why: only 3.2% of boxes are under 32 px at 640 px, so bigger images have little to rescue. 960 px has 2.25× the pixels per image, so **640 px is the choice**.

**Feasibility checks (AI-assisted, logged):** a model trained only on clinic photos scored 0.92 there but 0.09 on other photos. Adding other sources and healthy photos cut false alarms on Kaggle Normal from about 40% to 16%.

**Limits:** one run per size, so a 0.003 gap is noise · val/test hold copies of training photos (Aiden's finding), so 0.814 is optimistic · val/test are mostly Roboflow photos · Calculus is the weakest class (test mAP50 0.63), mixed up with caries and stains

---

<!-- SPEAKER NOTES (not on the slide) - about 3.5 minutes -->

## Speaker notes (~3.5 min)

**1. What my model does (~30 s)**
- The brief's first gap is detection: patients cannot tell caries or calculus from a normal mouth.
- My model is a YOLO detector. It draws a box around each problem and names it, so the patient sees where it is.
- The brief notes that Kaggle has one label per photo even when a mouth shows several conditions. A detector handles that; a single-label classifier cannot.
- Kaggle has no boxes and only 219 caries photos, so a detector cannot be trained on Kaggle alone. The team dataset adds public datasets that already have boxes.

**2. What I did on the dataset (~1 min 15 s). This is the main part.**
- I started with the Zenodo caries dataset: 6,265 clinic photos, 2,245 of them with caries boxes drawn by the dataset's annotators.
- Before training I checked it in my EDA notebook. Four things came out:
  - 4,038 photos have no label file. They are unlabelled, not healthy, so I left them out. Treating them as healthy would teach the model to ignore real caries.
  - 16 photos have boxes outside the image. I dropped them.
  - Two classes, decay on adult teeth and on baby teeth. I merged them into one caries class.
  - The authors' own train/val/test split puts 280 of 1,020 patients in more than one split.
- That last one is data leakage: the model is tested on mouths it already saw in training, so the score looks better than it is.
- It went further. The same person can have two IDs, one for photos with cheek retractors and one without, taken minutes apart. 293 such pairs crossed splits.
- So I split by person: 1,020 IDs became 390 person groups, and the check now finds no pair in both train and test.
- The leak had raised mAP50 by about 0.02 (0.944 → 0.924).

**3. Training and results (~1 min)**
- The team then moved to one shared dataset with five conditions: calculus, caries, hypodontia, mouth ulcer, tooth discolouration. About 32,000 photos, Kaggle included.
- I fine-tuned YOLO11n from pretrained weights instead of training from scratch: it already knows edges and shapes, so it needs less data and time.
- Settings: 100 epochs, batch 16, early stopping after 20 epochs without improvement, fixed seed.
- My first run was 50 epochs and reached 0.793, still rising, so I doubled it to 100 epochs.
- Result at 640 px on the held-out test set: mAP50 0.814, mAP50-95 0.425, precision 0.78, recall 0.76. Validation was 0.808, so choosing the best epoch on validation did not over-fit to it.
  - mAP50 is how well the boxes match the dentist's boxes, counting a box as correct if it overlaps the labelled box by at least half. 1.0 is perfect.
  - Recall 0.76 means it finds about 3 in 4 labelled problems. Precision 0.78 means about 4 in 5 of its boxes are right.
  - On only the person-labelled test images it scores the same, 0.815.
- Optimisation: image size 640 → 960 px, to help with small lesions. I changed only the image size; everything else was the same.
  - The 960 px run was trained on a Google Cloud GPU overnight, the 640 px run on my laptop.
  - 960 px learns faster: +0.022 mAP50 at epoch 25. But 640 px catches up: on validation 960 px ends only 0.006 ahead, and on the test set it is 0.003 behind (0.811 against 0.814).
  - My EDA explains it: only 3.2% of boxes are under 32 px at 640 px, so bigger images have little to rescue. 960 px helped calculus (+0.012) and caries (+0.004) a little and lost on hypodontia (−0.022).
  - 960 px images have 2.25× the pixels, so they are slower to train and to run in the app. With no gain, 640 px is the choice.

**4. Feasibility checks and limits (~45 s)**
- Before my own runs I used AI-assisted feasibility runs to test ideas quickly. They are logged separately and are not my submitted model.
- Main lesson: a model trained only on Zenodo clinic photos scored 0.92 on Zenodo but 0.09 on photos from another source, and flagged 44% of Kaggle Normal photos. It had learned one photo style, not caries in general.
- Adding other sources, healthy clinic photos and Kaggle Normal photos cut those false alarms to 16%, with about 95% of Kaggle caries photos still flagged. This matters because the brief says the tool is a screening aid: too many false alarms and people stop trusting it.
- Limits: it is one run per image size, so gaps of 0.003-0.006 could be chance. Aiden found that most val/test photos with a condition are copies of training photos, so my 0.814 is optimistic and I will re-score on his clean test. Val and test are mostly Roboflow photos, so the score says less about other photo styles. Calculus is the weakest class (test mAP50 0.63): the confusion matrix shows it mixed up with caries and discolouration, and it is the most common false alarm.
- Next: pick the app's confidence cut-off, and score on Aiden's clean test set.
