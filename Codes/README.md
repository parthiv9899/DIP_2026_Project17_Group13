# Codes: adaptive fabric defect detection and localisation (Group 13, Project 17)

Classical computer vision only (no deep learning). This continues the Week-7 Gabor + local texture
deviation pipeline, at full resolution (256 x 4096) and with adaptive, multi-scale parameters.

| File | What it is |
|---|---|
| `afid.py` | AFID listing, mask merging and dataset quirks (`python afid.py data` checks 140 + 105 strips) |
| `features.py` | Pre-processing, weave-period estimate, per-pixel filter bank, LBP / GLCM / Gabor window descriptors |
| `pipeline.py` | Feature cache, nested cross-validation by image, window + pixel models, thresholds, masks, metrics |
| `report.py` | Excel workbook, figures, overlays, deviation maps and masks in `Results/` |
| `run_all.py` | The whole pipeline in one command |
| `download_afid.py` | Downloads and extracts the three official AITEX archives into `data/` |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r Codes/requirements.txt
.venv/bin/python Codes/download_afid.py          # or symlink an existing AFID copy to ./data
.venv/bin/python Codes/run_all.py                # about 1 hour on 3 cores; resumes from cache/
```

`data/` must hold `NODefect_images/`, `Defect_images/` and `Mask_images/`. Caches go to `cache/`,
results to `Results/`. Three worker processes, one thread each (`WORKERS` in `pipeline.py`).

Self-checks: `python Codes/features.py` (synthetic weave: period and line detection) and
`python Codes/pipeline.py` (threshold search and metrics on a toy mask).

## Method

1. **Pre-processing**: illumination flattening (image minus a sigma-25 Gaussian), 3x3 median, CLAHE.
2. **Adaptive texture parameters**: the weave period along x and y is read from the row/column power
   spectrum of each strip; it sets the Gabor wavelengths per orientation and the GLCM distance.
   The fixed alternative uses the Week-7 Gabor wavelength (10 px, plus 20 px) and GLCM distance 1.
3. **Per-pixel filter bank** (robust z-score within the strip, median / MAD): deviation from a
   21x21 median, local std, row and column line profiles (broken picks / ends), LoG at sigma 2 and 4,
   adaptive Gabor energy, plus smoothed context maps.
4. **Window descriptors** on a 16-px cell grid with 32 and 64 px windows: intensity, LBP riu2
   (R=1 P=8, R=2 P=16) histograms, GLCM contrast / dissimilarity / homogeneity / correlation
   (0 and 90 degrees), Gabor energy, filter-bank mean and max. Each is strip-normalised (median/MAD
   over the strip), which removes the global capture shift between AFID defective and defect-free strips.
5. **Learning on hand-crafted features**: gradient-boosted trees give a defect probability per cell
   (window model) and per pixel (pixel model). Mask = window prob >= tw AND pixel prob >= tp, gated
   by the image decision; components under 10 px dropped; optional Hough line linking of
   near-horizontal / vertical segments for thin defects.
6. **Confidence**: isotonic-calibrated P(defect) and a split-conformal prediction set (alpha 0.1)
   from inner out-of-fold scores, giving a `reliable / uncertain / atypical` label per strip.
7. **Evaluation**: outer 5-fold CV by image (stratified by label and fabric); inner 3-fold CV chooses
   every threshold. Variants V0 to V4 move one step at a time from fixed to adaptive parameters;
   thresholds are global or per fabric.
