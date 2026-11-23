# Daniel’s check-in script

Keep the first slide up. The short version takes about 3 minutes. Reading the whole detailed version takes about 20–24 minutes, so you can also pull out individual sections when the lecturer asks about them. Bracketed lines are cues for you.

## If you have about 3 minutes

[Start on the left.]

I worked on the dataset and both models, then put them into a demo. The classifier flags photos that might show a problem. YOLO puts boxes around the things it finds. It can show more than one finding in a photo.

[Point to the dataset and Baidu line.]

I started by collecting more photos with a Baidu scraper. I used Chinese dental search terms because they gave me more actual mouth close-ups. I used MD5 to catch identical files and a 64-bit perceptual hash to catch edited copies. In the web pass, fingerprints up to six bits apart counted as a match. That took 21,000 downloads down to 12,968 photos. Only 665 passed Qwen’s check for a clear, usable mouth photo, and 406 made it into the final dataset after the other filters.

I used Qwen differently depending on what the source already told me. Roboflow already had human boxes. DentalAI and Zenodo had caries boxes, so I kept those and checked for the other conditions separately. A Kaggle folder gave me the condition, but I still needed a box. For web photos, all I had at first was the search term.

One mistake showed up when I looked at the classifier’s confident errors. It was flagging photos labelled normal, and I could see decay or staining in some of them. I found that my code treated some Zenodo photos with no annotation file as normal. I removed them and also removed web negatives that had only turned up because of a condition search. I checked Qwen’s boxes against human ones too. The average overlap was poor, so all the positive boxes in our main test came from human annotations.

[Point to the model cards, then the hospital card.]

MobileNet scored 0.944 macro-F1 on my separate classifier holdout, versus 0.975 on the main test. YOLO12m got 0.826 mAP50 on that main test. There’s a problem with the split, though. Every positive came from Roboflow, and every negative came from somewhere else. A model could use the way a photo looks to guess which set it came from.

The team then tested both models on the same 114 hospital photos they hadn’t trained on. MobileNet scored 0.475 AUROC, close to chance on that sample. YOLO scored 0.849, but it still flagged only 36% of the positive photos at the benchmark’s cutoff. That’s the result I’d use to decide what to do next. I’d get positives and negatives from the same clinics and test what happens when the whole app has to make a decision.

## Detailed speaking script

1. What I started with
[Open on the title. Skip to section 2 if the lecturer already knows the Kaggle work.]

I started with one Kaggle dataset, predicting one class per photo: five conditions plus normal, 4,375 files. Many were the same photo rotated, recoloured, or cropped — Calculus was worst: 1,296 files but only about 255 distinct photos. A random split would leak rotated copies between train and test, so I deduplicated first.

An MD5 hash on each file's bytes caught byte-for-byte copies regardless of filename; I reused that fingerprint as the filename for the resized cache. But re-saving as JPEG changes bytes, so I also used a 64-bit DCT perceptual hash (pHash): grayscale, resized to 32×32, Discrete Cosine Transform, keep the top-left 8×8 low-frequency block, take the median excluding the DC term (which tracks overall brightness), then binarise each of the 64 values against the median. Two photos match when their Hamming distance — differing bit positions — is ≤ 6.

I also embedded each image with an ImageNet-pretrained EfficientNet-B0 (classification head removed), L2-normalised the feature vector, and matched on cosine similarity ≥ 0.90 — this catches rotations, shears, and recolouring. I linked a pair if it passed either cutoff, then grouped by connected components: A–B and B–C keeps all three together even if A–C fails the cutoff.

Each photo group went entirely into one split, so no version could train while another was tested; validation and test use one photo per group. Groups whose copies carried conflicting class labels stayed in training, and the training sampler was balanced by class and by photo group so extra copies didn't overweight an original.

Every model started from ImageNet weights with a new six-output head (normal + five conditions), trained with cross-entropy on the single label per photo, full fine-tuning via AdamW including the pretrained layers.

I tested 12 backbones — ConvNeXt Tiny/Small/Base, EfficientNet B0/B3/V2-S, ResNet-50, RegNetY-3.2GF, DenseNet-121, MobileNetV3-Large, Swin-T, ViT-B/16 — 13 runs since ConvNeXt-Tiny was trained twice. Cap of 25 epochs with early stopping after six non-improving epochs, keeping the best checkpoint.

The baseline shuffled the file list and saw each file once, so a photo with ten copies appeared ten times. The other runs used a weighted sampler: equal total probability per class, equal share per photo group within a class (split among its files), draws per epoch equal to the training file count in batches of 32, with replacement — so some files repeat and others are skipped.

ConvNeXt-Tiny isolated the effect of the training setup with the model fixed. Baseline: horizontal flips only, 13 epochs, best at epoch 7. Second run: weighted sampler plus cropping, rotation, colour jitter, blur, JPEG compression, and label smoothing; ran all 25 epochs, best at epoch 23.

On original test photos, macro-F1 rose only 0.915 → 0.925 and caries recall held at 72%. The gap appeared when I degraded the same test photos to 160 px on the longest side at JPEG quality 60: macro-F1 fell to 0.553 for the baseline vs 0.886 for the stronger setup — normal-class recognition went from 1/40 to 35/40. Image quality had been doing some of the baseline's work. Sampling, augmentation, and label smoothing changed together, so this reflects the whole setup, not one factor.

The selection rule was fixed before training: average (equal weight) of macro-F1 on original validation photos and on the same photos degraded by low resolution, blur, poor lighting, or JPEG compression; take every run within 0.01 of the best average, then pick the fastest on CPU.

ConvNeXt-Base had the best average, 0.948, setting the cutoff near 0.938. EfficientNet-B0 scored 0.943 at ~9.4 ms/photo on CPU (ConvNeXt-Base: ~70 ms). MobileNetV3-Large was faster (~7.2 ms) but its 0.922 fell outside the cutoff. So EfficientNet-B0 qualified as the fastest: 20 epochs, checkpoint from epoch 14.

On the held-out test of 321 unique photos it scored 0.936 macro-F1, dropping to 0.836 under the low-resolution JPEG test. The test set was small, so we decided to get more data.

I also checked how much to trust the multiclass model’s confidence scores. I saved the raw scores, or logits, for the validation photos and kept the model weights fixed. I started with a temperature of 1. For each temperature, I divided the logits by it and measured cross-entropy against the true labels. Cross-entropy penalises giving the right class a low probability. I used L-BFGS to adjust the log of the temperature to minimise that loss. Taking the exponential kept the temperature positive. It came out to 0.592. Dividing the logits by that value before softmax adjusted the probabilities and kept the same top class. On the test photos, the calibration error dropped from 0.083 to 0.012.

Then I used the model’s highest class probability as its confidence for each validation photo. I checked cutoffs from 0.50 to 0.99 in steps of 0.01. At each cutoff, I counted how many photos were at or above it and how many of those the model got right. I took the first cutoff that gave at least 95% accuracy on the photos it answered and still answered at least half of all validation photos. The first cutoff, 0.50, met both goals. On the test set it answered 320 out of 321 photos and got 95.6% of those right.

### 2. How I scraped and cleaned the web photos

[Point to the Baidu line.]

I wanted photos that looked different from the Kaggle ones, so I scraped Baidu, Bing, and DuckDuckGo. Baidu was searched with Chinese clinical terms since it returned more useful images. I used multiple terms per condition plus variations like "close-up" or "clinical photo", targeting 3,000 images per category.

The scraper kept only files Pillow could read, with a shorter side ≥ 128 px, in JPEG/PNG/WebP/BMP; dropped duplicates by MD5; logged URL, engine, query, and dimensions to a CSV; and saved completed queries to a progress file so runs could stop and resume.

The downloader checked MD5 within each class folder. I then compared 64-bit DCT pHashes across all folders, as in the classifier: linked pairs ≤ 6 bits apart, grouped the links, and kept the largest-pixel-area image per group (ties broken by larger file size). That cut 21,000 downloads to 12,968 unique candidates. About 3,843 appeared under more than one condition — searching caries and discoloration can return the same photo — so I kept those under both classes.

Screenshots, diagrams, collages, and photos with tiny or obscured mouths remained, so I asked Qwen yes/no questions: real human teeth visible? clear and unobstructed? close-up photograph? collage? Only 665 passed — about 5% of unique candidates. After checking conditions and applying the remaining filters, 406 scraped photos made it into the 33,952-image build. (Wanted to do gingivitis but got lazy — couldn't confirm the issue or match images from existing datasets.)

### 3. How I used Qwen, and where it got things wrong

[Stay on the dataset card.]

I ran Qwen3.8 flash next locally on my personal server. Roboflow already had human boxes for all five conditions, so those needed no Qwen call. DentalAI and Zenodo had caries annotations — I kept those and asked Qwen only for the other four findings. Kaggle gave the class in the folder name, so Qwen just boxed the known condition. Mendeley was a non-caries dataset: I left caries out of the prompt and checked for the other targets. Scraped photos were the hardest — Qwen had to identify which condition was visible before boxing it.

For photos with no known condition, I separated naming a finding from drawing its box. The first answer returned the evidence, the label, and a confidence (high/medium, used only for filtering); the next request asked for a box around the finding it had just described. The model returns four corner coordinates on a 0–1000 scale (a method tested and documented by Alibaba); the notebook converts them to YOLO format — class ID, centre x/y, width, height, all as image fractions.

The first prompt over-called caries. I added concrete criteria: deposit stuck to the tooth suggests calculus; visibly missing tooth structure suggests caries; a colour change on a smooth surface suggests discoloration. I told it to watch for fillings and shadows, and kept only high-confidence findings when the model supplied the label. In a 130-image prompt comparison, false caries flags fell from 48% to 3% on the Kaggle normal subset and 37% to 8% on the caries-free Mendeley subset.

Box placement was a separate problem. On 32 photos versus human boxes, mean IoU — shared area divided by union area — was 0.19. I checked the coordinate conversion for scale or offset bugs; it was correct. Inspecting the images showed Qwen simply placed some boxes in the wrong places.

I built a box viewer showing the photo, its boxes, its source, and its original label, so I can click through examples and record notes like "box too big" or "wrong spot" (haven't done yet). From the patterns I saw I wrote rules: drop model boxes covering over 60% of the photo; drop discoloration boxes wrapped around an existing human caries box; drop human boxes thinner than 0.5% of the image — about three pixels at 640 px. Where Kaggle and Roboflow had the same photo, I kept the human-boxed copy.

### 4. The two label fixes that came from looking at mistakes

Using this dataset, I trained a new binary classifier (normal vs. any issue).

Sorting the earlier classifier's errors by confidence, some "normal" photos clearly showed decay or staining. I found two ways I'd created those bad labels. First, many Zenodo photos were never annotated, and my pipeline treated a missing annotation file like a checked photo with an empty one — I excluded about 4,000 never-annotated photos. Second, for web photos, my first version could mark a photo normal just because Qwen couldn't confirm a finding on it from a condition search; I dropped 75 such photos. A web photo now stays normal only if every search term that retrieved it was a normal search and it still passed the checks.

Cross-source macro-F1 at the 0.5 cutoff rose from 0.884 (first binary classifier) to 0.950 after the label cleanup. The backbone and test photos also changed between versions, so this isn't a clean measure of the label fix alone — that would need those held fixed.

The second issue began with a photo of an obviously missing front tooth that my test demo scored normal. In training, most missing-tooth examples came from Roboflow, where the gap reads as a very dark hole; in an ordinary photo, a gap can show pink gum.

From the first Qwen pass I'd seen it call teeth "missing" when they were angle-hidden or outside the crop, so I dropped Qwen's self-guessed missing-tooth boxes — 258 in the saved filter run. I kept human boxes and still allowed Qwen boxes on photos already filed under Hypodontia in Kaggle. A photo with a box for another condition stayed in YOLO; if the filter removed its last box, the photo left YOLO at that stage.

I revisited the 2,013 photos with a Qwen missing-tooth finding and asked one narrower question: is a tooth-sized gap visible inside the row in this picture? That kept 143 photos; 109 had no other boxes, so they went into the classifier as issue photos. For YOLO, Qwen drew each gap from its own description of its location; I then sent the photo back with the box drawn in green and asked what was inside it. It drew 144 boxes; after size/shape checks and Qwen re-checking the marked photo, I kept 107 boxes on 106 photos. That final check was still Qwen — I sampled it, but the boxes aren't all human-verified.

I added gap boxes where possible, held 17 photos out of YOLO for checking, and routed photos with unusable boxes to the classifier. In the early comparison, YOLO detections on 46 held-out non-Roboflow missing-tooth photos went from 1 to 5; the classifier went from 4 to 5 of 13 in its own small check. The extra data helped a little, and both tests showed how often the models still miss these photos.

Potential fixes: human labelling; Qwen4 flash when it releases; giving it measurement/analysis tools; enabling thinking.

### 5. How I set up the split and trained MobileNet

[Point to “Source bias”, then the classifier card.]

Merging all six sources, I ran another pHash pass with the cutoff raised to eight bits — some Roboflow colour variants were farther apart — and also linked Roboflow exports sharing the same original filename before the .rf. suffix. Any group exceeding 50 photos stayed in training, limiting how far long match chains could affect the test set.

I also stopped ordering splits by temporary group number, which reshuffled old negatives as images were added. Instead, each group sorts by MD5 of "seed 42" + its smallest file MD5 — a repeatable ordering. The switch changed the split once; after that, adding the missing-tooth photos moved only two existing test photos in my check. (The team's later checks found edited copies my grouping missed.)

The combined build has 33,952 photos; YOLO uses 31,909, with 26,032 for training, and empty-label backgrounds capped at ~10% of the YOLO training list. The classifier still uses the extra negatives and issue photos without boxes. Main validation and test have human-drawn positive boxes, but most negatives were Qwen-screened (sometimes alongside human caries annotations), so I kept a fully human-labelled test subset — all 2,315 positives, but only 59 negatives.

The main test-composition risk: every positive comes from Roboflow, every negative from elsewhere — and Roboflow's consistent export size and recognisable style let a model win by learning the source. So for the classifier I held out two photo-group sets from the non-Roboflow pool — 1,181 images each — excluded from training.

Positives far outnumbered negatives and copy counts varied, so the sampler draws the two labels ~equally; within positives, ~half the draws go to Roboflow and half to other sources; within buckets, a photo's copies share one group's weight. Each epoch draws 12,000 samples. I used pretrained weights, a two-output head, AdamW fine-tuning. Evaluation resizes the whole image to 224×224 so an ulcer at the lip isn't cropped out; training varies crop, blur, JPEG quality, and lighting, with only small hue shifts since colour carries signal. Label smoothing is on because some labels came from Qwen.

I compared eight runs, capped at 15 epochs. Each epoch scores the mean of macro-F1 over clean validation, a simulated phone-damage copy, and the cross-source validation set; four non-improving epochs stop the run. Best score ~0.968; MobileNet reached ~0.965 — within 0.01 — at 8.3 ms/image on our CPU benchmark, so by my pre-stated rule (fastest within 0.01 of top), MobileNet won, finishing all 15 epochs with its best checkpoint at epoch 12. On the synthetic blur test, the ConvNeXt baseline fell to 0.854 macro-F1 while the stronger recipe on the same backbone held 0.943.

The slide numbers, 0.975 and 0.944, both use a 0.5 cutoff. Macro-F1 gives normal and issue equal weight. For the app, I used the same temperature-scaling method as for the multiclass model. I joined the logits and labels from the main and cross-source validation sets, then fitted one temperature by minimising cross-entropy on those photos. That gave me 0.706.

For the binary cutoff, I took the calibrated probability of an issue and looked at the photos actually labelled issue. On each validation set, I sorted their probabilities from lowest to highest. The goal was to miss at most 5%, so I counted how many that let me miss, rounded down, and used the next score as the cutoff. A photo right at the cutoff still counted as flagged. For example, cross-source validation had 606 issue photos. I could miss at most 30, so I used the 31st-lowest score. The main validation cutoff was 0.974, and cross-source validation gave 0.729. I took the lower cutoff so both met the 95% target. On the cross-source test it caught 92.8% of positives and left 96.3% of negatives unflagged. It caught only 1 of 8 Mendeley positives, so I also checked the results by source.

### 6. How I trained and picked YOLO

[Point to the YOLO card.]

YOLO learns to predict a class and a box for each finding. I fine-tuned from COCO-pretrained weights at 640 pixels. Some source images were over 8,000 pixels across and costly to decode, so I made cached copies with a longest side of 1,024 pixels. I used batches of 32 for the smaller YOLO models and 16 for the medium models to fit GPU memory. Ultralytics accumulated gradients across batches to work towards 64 images per optimiser update.

I left the YOLO optimiser on auto. Ultralytics picked MuSGD, which combines Muon with stochastic gradient descent. For YOLO12m, the base learning rate was 0.01, with a higher rate of 0.03 for the class-prediction layers. I used three epochs of warmup, then decreased the learning rates linearly. Weight decay was 0.0005. RT-DETR-L used AdamW with a learning rate and weight decay of 0.0001 and a batch size of 8.

YOLO12m’s training loss combined box overlap, classification and distribution focal loss, which helps place the edges of each box. The weights for those three terms were 7.5, 0.5 and 1.5. For augmentation I used mosaic, which combines four training images, along with random scaling, shifts and horizontal flips. I kept colour changes small because colour helps identify calculus and discoloration. I disabled vertical flips and rotation.

I used the same validation split to compare the nine completed runs, including YOLO11, YOLO12, YOLO26 and RT-DETR-L. I capped training at 80 epochs and stopped a run after 15 epochs without improvement on the trainer’s validation score.

I chose by validation mAP50–95. I’d only use compute as a tie-break if two runs were within 0.005. YOLO12m got 0.463, YOLO26m got 0.453, and the previous YOLO11m got 0.452. YOLO12m’s lead was big enough that I kept it even though it was slower. It ran for 66 epochs, and its best validation mAP50–95 was at epoch 51. On the main test it got 0.826 mAP50 and 0.444 mAP50–95. For mAP50, the class has to match and the box has to reach 0.50 IoU. The second score averages from 0.50 up to 0.95, where a box has to line up much more closely.

I also ran YOLO11-small twice to see what the Qwen images were doing. One version trained on 18,664 fully human-labelled images. The other trained on 26,032 images, including the Qwen-assisted ones. On the same human-labelled test, mAP50–95 was 0.428 versus 0.431. That’s a small difference. The bigger set changes both how many examples the model sees and which examples it sees, so it doesn’t measure label quality by itself. (About same performance so Qwen is kept)

I also looked at how well YOLO named and located each condition. Calculus was the weakest at 0.666 AP50, and mouth ulcers scored highest at 0.927.

### 8. What I put in the demo and what I’d change

[Finish on the left panel.]

I put the models behind a FastAPI demo. First I check that the upload has a usable mouth. CLIP decides whether it looks like a mouth close-up, a face or something else. If it’s a face with an open mouth, YuNet gives me the mouth corners so I can level and crop it. Then MobileNet scores the photo. If it flags it, YOLO looks for boxes on that same photo or crop. When the detector has no confident box, the demo can show dashed candidates as hints.

I built SmileCheck SG on top of that so I could try the same idea with multiple photos and video. The backend sends progress as frames are processed, and the page can show the frame with the strongest finding. A video normally needs at least two flagged frames and at least 10% of its readable frames to flag the whole clip. The overall result is flagged if any photo or video qualifies. The prototype also has a detector-only switch, explanations for each condition and a clinic map with sample entries.

I also tried an offline version where both models scored every photo and a hit from either model counted. That was with the older YOLO11m. The classifier missed 15 positives and YOLO missed 25 on the main test. Each missed photo was caught by the other model, so the combined rule caught all 2,315 positives and flagged 38 of 621 negatives. I’d treat that as a reason to try the rule again. The app usually runs YOLO only after a classifier flag, and that test had the source issue I mentioned earlier.

My next test would run four versions on the same unseen photos. I’d try the classifier alone, YOLO alone, the classifier as a gate, and both models on every photo. I’d compare how many issue photos each misses, how many negatives it flags, whether the boxes are on the right findings, and how long it takes. I’d want positives and negatives from the same clinics or devices, patient-level splits, and agreement on what the boxes should cover. For use in Singapore we still need local patient validation. The app currently covers five findings, and gum disease is outside that list.

## If the lecturer asks for more

**“What was in each source?”**

| Source | What I started with | Photos in the final build |
|---|---|---:|
| Roboflow Oral | Human boxes across all five target conditions | 23,277 |
| DentalAI | Human caries annotations | 2,439 |
| Zenodo | Human caries annotations on part of the dataset | 2,228 |
| Kaggle Oral | Condition names as folders | 993 |
| Mendeley Noncaries | Evidence for absence of caries | 4,609 |
| Baidu | Search terms and source URLs | 406 |

**“How did you make the long jobs repeatable?”** I saved answers to JSONL as Qwen finished each step. I could stop halfway, restart from the terminal and use the answers I already had. I applied the confidence and box rules after reading those answers, which let me change a filter without calling Qwen on everything again. The final manifest kept the hash, source, label provenance, group, split and reasons for dropping files. I wrote the dataset zip to a temporary file first and renamed it when it was ready. I cached smaller classifier images, moved the dataset class into an importable Python file for Windows worker processes, and saved model checkpoints so I could reuse or resume runs.

**“What was the difference between the two classifier holdouts and the outside benchmark?”** My cross-source validation and test came from sources already in the combined dataset. I set aside whole photo groups for the classifier. The team benchmark used outside data and checked for copies against the team’s training photos, including with perceptual hashes and SIFT. That benchmark was a better check of how these models deal with another source. The internal cross-source result is still handy for checking changes to the classifier.

**“How did you check whether the source shortcut was real?”** I can show which sources land on each side of the split. In the main evaluation, every positive was Roboflow. Brandon also tested a simple rule based on whether the image was 640 by 640 and showed it could exploit the split. Aiden later checked for edited copies using image features and SIFT. I looked at the classifier’s high-confidence errors and Grad-CAM overlays for individual examples. A heatmap can help me decide what to inspect next. To measure the shortcut, I’d use the separate test data.

**“If Qwen’s boxes had poor overlap, why keep them?”** The prompts gave me access to more kinds of photos, and I could still keep the image-level label when the box didn’t pass a check. For detection I used the boxes as extra training data, filtered the worst cases and tested on human-drawn positive boxes. The YOLO11-small comparison showed a small gain on that human-labelled test with the extra images. That’s about as far as I’d take the claim.

**“Did the filters improve Qwen’s IoU?”** I replayed the oversized-box rule on cached Qwen predictions for 1,369 DentalAI photos with 5,770 human caries boxes. I kept every human box in the scoring. For each one, I took the best IoU with a Qwen caries box and counted zero if Qwen had no box. The mean was 0.1783 both ways because none of the Qwen boxes in this sample covered over 60% of the image. This was a separate check from the original 32-photo audit. The rule only deletes boxes. It can cut out bad predictions, but with the same set of human boxes it can only leave the best overlap unchanged or make it lower. On a separate replay of Qwen’s extra-condition check, the final overlap rule removed 80 boxes that covered more than half of a human caries box. DentalAI only gave us caries references, so I can count those overlap removals, and I’d need human boxes for the other conditions to score them.

**“Did the either-model rule work outside the main test?”** In that older YOLO11m experiment, I checked 529 photos that neither model trained on. Only 11 had an issue. MobileNet flagged four, YOLO flagged two, and the combined rule flagged five. That’s too few positives to pick a rule for the app. A different group of 1,833 cross-source photos was already in YOLO’s training data, so I keep those numbers separate.

**“Did you train RF-DETR too?”** The saved comparison has RT-DETR-L, which got about 0.431 validation mAP50–95. I built a separate RF-DETR runner to convert the dataset layout and keep its dependencies separate from the CLIP demo. RF-DETR doesn’t have a completed result in the saved nine-run table.

**“What do you mean by missing teeth?”** In the dataset the class is called hypodontia. What I’m actually asking the models to recognise in the photo is a visible gap where a tooth would be. I can’t tell from the photo alone whether the tooth never developed, was removed, or was lost.

**“Where would you look first if we try a live demo?”** The FastAPI demo is now under `Codex/`. The SmileCheck SG loader still points at the old location. I’d check that path and the checkpoint paths, then try one close-up, one face crop and one rejected image. Still photos are handled in memory. The video code uses a temporary file for decoding and removes it afterwards. The clinic entries and cost section are prototype content.

### Epochs for each saved run

The classifier columns show **epochs run / best epoch**. The multiclass limit was 25 with patience 6. The binary limit was 15 with patience 4.

| Backbone and training setup | Multiclass | Binary |
|---|---:|---:|
| ConvNeXt-Tiny, baseline | 13 / 7 | 10 / 6 |
| ConvNeXt-Tiny, balanced | 25 / 23 | 9 / 5 |
| ConvNeXt-Small | 25 / 24 | n/a |
| ConvNeXt-Base | 24 / 18 | n/a |
| EfficientNet-B0 | **20 / 14** | 14 / 10 |
| EfficientNet-B3 | 14 / 8 | n/a |
| EfficientNetV2-S | 25 / 22 | 15 / 13 |
| ResNet-50 | 16 / 10 | 15 / 13 |
| RegNetY-3.2GF | 10 / 4 | n/a |
| DenseNet-121 | 19 / 13 | 15 / 15 |
| MobileNetV3-Large | 18 / 12 | **15 / 12** |
| Swin-T | 21 / 15 | 9 / 5 |
| ViT-B/16 | 20 / 14 | n/a |

The detector training limit was 80 with patience 15. The best epoch here is the one with the highest validation mAP50–95 in the saved history.

| Detector | Epochs run | Best epoch |
|---|---:|---:|
| YOLO11n | 80 | 80 |
| YOLO11s | 78 | 62 |
| YOLO11m | 69 | 59 |
| YOLO11s, human labels only | 57 | 42 |
| YOLO12s | 64 | 51 |
| YOLO12m | **66** | **51** |
| YOLO26s | 66 | 46 |
| YOLO26m | 66 | 51 |
| RT-DETR-L | 48 | 33 |

For the multiclass weighted sampler, each file’s weight was `1 / (number of photo groups in its class × number of files from its class in that photo group)`. The number of replacement draws per epoch was `len(train_df)`. The notebook’s run log shows 3,136 training files and the committed Kaggle manifest lists 3,130, so the rule is the clearest way to describe the configuration.

### Numbers to have handy

| If asked about | Say |
|---|---|
| Original classifier | EfficientNet-B0, 321 unique Kaggle test photos, macro-F1 0.936 [0.900, 0.965] |
| Multiclass confidence | Temperature 0.592, top-class confidence cutoff 0.50. Answered 320 of 321 test photos with 95.6% accuracy on those answered |
| MobileNet headline numbers | 0.975 main and 0.944 cross-source macro-F1 at cutoff 0.5. Cross-source 95% interval [0.929, 0.957] |
| MobileNet in the app | Temperature 0.706, cutoff 0.729. Sensitivity / specificity 99.2% / 94.7% main test, 92.8% / 96.3% cross-source |
| YOLO12m main test | 0.826 mAP50 and 0.444 mAP50–95 across 2,936 photos |
| YOLO12m AP50 by class | Calculus 0.666, caries 0.820, missing-tooth gaps 0.813, ulcers 0.927, discoloration 0.901 |
| Outside hospital photos | MetaDent DS1, 94 positive + 20 negative. AUROC 0.475 [0.34, 0.62] for MobileNet and 0.849 [0.75, 0.92] for YOLO12m |
| Outside caries boxes | AlphaDent, 518 photos with 1,527 caries lesions. YOLO12m AP50 0.023 and AP10 0.107 |

## Where the details came from

- Data collection and labelling. [Scraper](../../models/classifier/scripts/Daniel/scrape_web_images.py), [Qwen notebook](../../models/classifier/notebooks/Daniel/qwen_clean_and_label.ipynb), [box viewer](../../models/classifier/scripts/Daniel/view_yolo_boxes.py), [work summary](daniel_work_summary.md)
- Original Kaggle experiments. The six-class notebook is in commit `4fae207` at `models/classifier/notebooks/Daniel/oral_condition_classifier.ipynb`. Deduplication is in cells 6, 13 and 14 (zero-based). [Kaggle experiment table](../../models/classifier/results/Daniel/experiments.csv)
- Classifier training and checks. [Classifier notebook](../../models/classifier/notebooks/Daniel/oral_condition_classifier.ipynb), [experiment comparison](../../models/classifier/results/Daniel/binary/experiments.csv), [saved metrics](../../models/classifier/results/Daniel/binary/test_metrics.json), [split manifest](../../manifests/Daniel/binary_classifier_split.csv)
- Detector training. [YOLO notebook](../../models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb), [YOLO12m training history](../../models/yolo_caries/results/Daniel/history/yolo12m.csv), [Ultralytics optimiser implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.211/ultralytics/engine/trainer.py#L1112-L1198)
- Detector comparisons. [Validation comparison](../../models/yolo_caries/results/Daniel/comparison_val.csv), [YOLO12m metrics and training settings](../../models/yolo_caries/results/Daniel/test_metrics.json), [older YOLO11m comparison](../../models/classifier/results/Daniel/binary/clf_vs_yolo/analysis.txt)
- Independent checks. [Team decisions](../decisions.md), [benchmark procedure](../../benchmark/README.md), [external benchmark results](../../benchmark/results/benchmark_report.md)
- Demo and prototype. [FastAPI demo](../../models/classifier/demo/Daniel/Codex/app.py), [mouth gate](../../models/classifier/demo/Daniel/Codex/mouth_gate.py), [SmileCheck SG backend](../../prototype/Daniel/server.py)
