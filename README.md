# TKI Treatment ML

Drug-response modeling of organoids from multi-channel microscopy: segmentation,
feature extraction, and leave-one-out modeling, with the processed tables needed
to reproduce the modeling results.

## Layout

```
src/            pipeline and utility scripts
data/
  raw/          raw image trees (not included)
  masks/        one binary mask per well (CH4 segmentation)
  processed/    cohort tables, extracted features, deep embeddings
results/        modeling outputs
requirements.txt
```

Raw images are not included. `cohort_audit.py` and `segment_cohort.py` read them
from `data/raw/`; everything downstream runs from `data/processed/` and
`data/masks/`.

## Pipeline

```
cohort_audit.py       build the cohort table from data/raw/
segment_cohort.py     SAM2 segmentation on the CH4 brightfield channel, one mask per well
extract_features.py   23 features per channel over the largest mask component
deep_features.py      frozen DINOv2 embeddings per well (ResNet50 fallback)
train_loo.py          leave-one-out modeling on features_v2.csv (label_drug)
```

`train_loo.py` subtracts each well's dataset baseline (untreated mean), then
z-scores within dataset; every fitted step is refit inside the LOO fold. Outputs
go to `results/`.

## Utilities

```
image_inventory.py         image and channel inventory
fluor_alignment_check.py   fluorescence / mask alignment check
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA 12.1 PyTorch build; drop the extra index URL and
the `+cu121` tags for CPU. `tabpfn` and `sam2` are only needed for the modeling
and segmentation steps.
