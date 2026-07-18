# Topic Modelling Internship Work

This repository organizes NLP internship work around topic modelling and review analysis.

## Source

Original Google Drive folder:

https://drive.google.com/drive/folders/1XfbC_M6OK70448Hpm130Vstqm69WH3HG

Additional internship work folder:

https://drive.google.com/drive/folders/1AFg-GrGi47McAk9mDwD9dM-tb6S8hMvN

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

Source scripts, the EDA report, the classifier workbook, and the UMAP figure were imported locally from Google Drive and committed to the repo.

Very large Drive artifacts are intentionally not tracked in normal Git because GitHub rejects files over 100 MB. Keep these in Drive or add Git LFS before importing them:

- `summary doms job review.mp4`
- `summary of DOMS job review.zip`
- `updated_glassdoor_data.csv`
