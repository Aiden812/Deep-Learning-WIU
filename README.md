# Deep-Learning-WIU

Oral-health screening from intraoral photos: an image classifier for common conditions, a YOLO caries detector, an image-quality check, and a GenAI layer (RAG + agent) that explains results and estimates treatment costs with Singapore subsidy schemes.

## Repository layout

| Path | Purpose |
|---|---|
| `shared/conditions.yaml` | Class IDs, names, urgency, procedures — single source of truth |
| `data/` | Datasets (gitignored). `raw/` → `interim/` → `processed/` |
| `manifests/` | Split CSVs and file hashes, so splits are reproducible without committing data |
| `models/classifier/` | Multi-class condition classifier |
| `models/yolo_caries/` | Caries object detector (YOLO) |
| `models/quality_check/` | Rejects blurry / dark / off-target photos |
| `genai/rag/` | Retrieval over HPB / NDCS material (`corpus/` must be sourced + dated) |
| `genai/agent/` | Agent tools: CHAS clinics, products (sourced + dated) |
| `genai/guardrails/` | Input/output safety checks |
| `cost/` | Procedure → cost mapping and subsidy calculator |
| `app/` | Backend + frontend |
| `deploy/` | Remote deployment (PR2) |
| `tests/` | Automated tests |
| `docs/` | Decisions, test log, ethics checklist, data flow, AI-use logs |

## Datasets

Download into `data/raw/<name>/` (see [Dataset links.md](Dataset%20links.md)):

| Folder | Source |
|---|---|
| `kaggle_oral/` | https://www.kaggle.com/datasets/jhirshaikh/oral-image-dataset |
| `zenodo_benchmark/` | https://zenodo.org/records/14827784 (benchmarking split) |
| `zenodo_full/` | https://zenodo.org/records/14827784 (full dataset) |
| `dentalai/` | https://datasetninja.com/dentalai |
| `mendeley_noncaries/` | https://data.mendeley.com/datasets/6zsnhrds9t/1 |

Then run `python models/yolo_caries/scripts/check_downloads.py` to verify.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r models/yolo_caries/requirements.txt
cp .env.example .env            # fill in keys
```
