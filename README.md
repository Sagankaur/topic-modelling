# Topic Modelling Internship Work

This repository organizes NLP internship work around topic modelling and review analysis.

## Source

Original Google Drive folder:

https://drive.google.com/drive/folders/1XfbC_M6OK70448Hpm130Vstqm69WH3HG

## Layout

- `src/` - Python scripts for cleaning, EDA, classification, topic modelling, reporting, and dashboards.
- `data/raw/` - source or large input datasets.
- `data/processed/` - cleaned and enriched datasets.
- `outputs/` - generated topic-model outputs, summaries, spreadsheets, and images.
- `reports/` - written analysis and documentation.
- `archive/drive-folders/` - original folder-level outputs preserved from Drive.
- `docs/drive_manifest.csv` - import map from the original Drive folder.

## Environment

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The original environment used CUDA-enabled PyTorch packages (`torch`, `torchaudio`, and `torchvision` with `+cu121`). If installation fails on a non-CUDA machine, install the PyTorch build that matches your system from the official PyTorch selector, then install the remaining requirements.

## Import Status

The Drive connector can see the folder, but direct local download via `gdown` returned HTTP 401. To complete a full local import, temporarily set the Drive folder to **Anyone with the link: Viewer** or provide a ZIP export, then download into this repo and place files according to `docs/drive_manifest.csv`.
