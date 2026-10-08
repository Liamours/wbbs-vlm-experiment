"""Project folder layout. This repo sits at <project>/repo/wbbs-vqa_vgrounding; datasets and models sit in <project>.

Set WBBS_PROJECT_ROOT to use a different project folder, for example a standalone clone of this repo.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("WBBS_PROJECT_ROOT") or Path(__file__).resolve().parents[3])
DATASETS = PROJECT_ROOT / "datasets"
MODELS = PROJECT_ROOT / "models"
BASE_MODELS = MODELS / "base"
SOURCES = DATASETS / "sources"
AUDITS = PROJECT_ROOT / "results/analyses/dataset-audits"

# Base weights live next to the adapters under models/, not in the user cache; set before any Hugging Face import.
os.environ.setdefault("HF_HUB_CACHE", str(BASE_MODELS / "hub"))

# METEOR needs WordNet; keep it in the project instead of the user folder.
os.environ.setdefault("NLTK_DATA", str(SOURCES / "nltk_data"))
