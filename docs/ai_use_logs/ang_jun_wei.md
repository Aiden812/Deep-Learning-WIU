# AI use log — Ang Jun Wei

| Date | Tool | What you asked for | Accepted / changed / rejected | Why – what you checked and what you found |
|---|---|---|---|---|
| 3 Nov | Claude Code | Explain the section 1 output (6,265 images, 2,245 labels, 4,038 without labels) | **Changed** | Claude first said the 4,038 images were "no caries". I asked whether that was true. The authors' COCO files list only the 2,227 labelled images, and the Zenodo page never says the unlabelled ones are healthy. Changed the notebook to "unannotated" and excluded them. |
| 3 Nov | Claude Code | Which train/val/test split to use | **Rejected** first advice | Claude first suggested the authors' benchmark split. A check by patient ID found 280 of 1,020 patients in more than one split, which is data leakage. Decided to split by patient myself; added a cell that shows the leak. |
| 3 Nov | Claude Code | Explain why section 5 reports 760 pilot images without an ID | **Changed** | Section 5 counted all images, including the unlabelled ones it doesn't use. Changed `imgs` to `imgs & labs`. The numbers changed from 760 to 270 without an ID, and from "most patients have 5 photos" to "most have 2". Updated the Finding and the summary. |
