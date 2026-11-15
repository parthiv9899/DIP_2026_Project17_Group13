# Results: Group 13, Project 17 (AFID, classical texture descriptors)

All numbers are on held-out strips. Every one of the 245 usable AFID strips (140 defect-free,
105 defective) is tested exactly once, by models and thresholds that never saw it.

## Protocol

- **Splits by image.** Outer 5-fold cross-validation, stratified by (label, fabric). Never split by window or pixel.
- **No tuning on test.** In each outer training set, a 3-fold inner CV produces out-of-fold scores. These choose:
  - the image threshold
  - the window and pixel thresholds
  - the isotonic calibrator
  - the conformal reference scores
  - the inner-CV model selection

  The test fold is used once, with everything frozen.
- **Pre-declared final model:** `V3_adaptive`, with global thresholds and the `post` mask (components under 10 px dropped). It was fixed before any test result was seen. The Hough-linked mask and per-fabric thresholds are reported as ablations.
- **Undefined ratios are NaN**, never 0. For example, precision on a strip with no predicted pixels.
- **Pixel metrics** are reported both pooled (TP/FP/FN summed over strips) and as per-image means. Tolerance metrics (±2 and ±3 px) are included because thin yarn defects are 2 to 6 px wide.
- **Capture shift.** Defective and defect-free strips of fabrics 00/01/02/05 differ globally. A classifier can exploit this without seeing the defect, so fabrics 03/04/06 are reported separately as the fair image-level test.

## Headline numbers

| Image level (245 strips) | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| V3_adaptive (pre-declared) | **0.776** | 0.784 | 0.657 | 0.715 | 0.843 |
| Inner-CV selected variant per fold | **0.804** | 0.777 | 0.762 | 0.769 | 0.876 |
| Inner-CV selected, fair fabrics 03/04/06 (107 strips) | **0.832** | 0.774 | 0.872 | 0.820 | 0.911 |

| Localisation, V3_adaptive, pooled over the 105 defective strips | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Pixel, exact | 0.496 | 0.285 | **0.362** | **0.221** |
| Pixel, with Hough line linking | 0.440 | 0.334 | 0.380 | 0.234 |
| Pixel, ±2 px tolerance | 0.577 | 0.347 | 0.434 | |
| Pixel, ±3 px tolerance | 0.605 | 0.366 | **0.456** | |
| Window (16 px cell) | 0.401 | 0.375 | 0.387 | 0.240 |

Pooled over all 245 strips, pixel F1 is 0.353 and IoU is 0.214. Per-image means over defective strips are lower: F1 0.183, IoU 0.117 and ±3 px F1 0.339. Many tiny defects score close to 0, and each one counts as much as a long broken yarn.

For comparison, Group 13's earlier baselines were:
- Week 7 (Gabor and local texture, 1024x256, global threshold): pixel F1 0.0002 and 0.0004, image AUROC 0.50.
- Mid-sem (dark-contrast detector): pixel F1 0.031, IoU 0.016.

## Fixed vs adaptive texture parameters

Global thresholds, from `Ablation_Fixed_vs_Adaptive`:

| Variant | Image acc. | Fair acc. | Window F1 | Pixel F1 | Pixel IoU | ±3 px F1 |
|---|---|---|---|---|---|---|
| V0 fixed: 32 px window, Week-7 Gabor lambda 10/20, GLCM d=1, raw | 0.763 | 0.785 | 0.336 | 0.306 | 0.181 | 0.404 |
| V1 + multi-scale windows (32 + 64 px) | 0.808 | 0.813 | 0.355 | 0.340 | 0.205 | 0.458 |
| V2 + adaptive Gabor wavelength and GLCM distance (weave period) | 0.812 | 0.776 | 0.367 | 0.332 | 0.199 | 0.427 |
| V3 + strip-normalised features (median/MAD), final | 0.776 | 0.822 | **0.387** | **0.360** | **0.220** | **0.463** |
| V4 raw + strip-normalised | 0.865 | 0.860 | 0.368 | 0.342 | 0.206 | 0.428 |

- **Fixed to adaptive improves localisation.** Window F1 goes from 0.336 to 0.387 and pixel IoU from 0.181 to 0.220.
- **Strip normalisation (V3) gives the best localisation**, and better fair-fabric accuracy than the raw-feature variants V0 to V2.
- **V4 has the highest image accuracy (0.865) when looked at after the fact.** The inner CV picked it in only 2 of 5 folds, so the honest image-level figure is the inner-CV selected 0.804. Reporting 0.865 as the result would be selecting a model on the test set.
- **Per-fabric (adaptive) thresholds** raise image accuracy for V0 and V2 but lower pixel F1 everywhere: 0.362 drops to 0.292 for V3. With 1 to 38 defective strips per fabric, per-fabric thresholds overfit.

## Per fabric and per defect

Final model, from `Per_Fabric` and `Per_Defect`.

Detection works well on thread defects:
- broken yarn: 8/8 detected, pixel F1 0.51, ±3 px F1 0.79
- broken end: 8/9, pixel F1 0.46
- broken pick: 8/10, pixel F1 0.55
- crease: 5/5
- weft curling: 3/3

It is weak on small blobs:
- fuzzy ball: 20/39 detected, pixel F1 0.09
- nep: 8/14
- warp ball: 3/5
- cut selvage: 4/9 (it sits at the fabric edge)

Fabric 06 reaches 0.96 accuracy and fabric 04 0.87. Fabric 00 is the weakest at 0.63.

## Confidence and reliability score

Each strip gets an isotonic-calibrated `P(defect)` and a split-conformal prediction set (alpha = 0.1), both from inner out-of-fold scores.
- **Reliable:** 55 % of strips get a single-label set. Their accuracy is 0.830, against 0.709 on the uncertain ones.
- **Calibration** is reasonable: strips with P(defect) of 0.8 to 1.0 are defective 86 % of the time, and strips with 0.0 to 0.2 are defective 19 % of the time.

## Where it falls short of the targets, and why

- **Image accuracy is 0.78 to 0.80, not 0.90.** Most defects are tiny blobs: fuzzy balls, neps and warp balls make up 58 of the 105 defective strips and are often 5 to 15 px across. On a 1-megapixel strip they are hard to tell apart from normal slubs, and recall on these classes is about 0.5. The thread defects (yarn, end, pick, crease) are detected at 80 to 100 %. Image accuracy on fabrics 03/04/06 is 0.83, and 0.86 to 0.96 on fabrics 04 and 06.
- **Pixel IoU is 0.22, not about 0.9.** This is expected for classical methods on 2 to 6 px structures with hand-drawn masks: a 1 px offset already halves the IoU of a 2 px line. With ±3 px tolerance, F1 is 0.46. Thread defects reach ±3 px F1 of 0.63 to 0.79.
- **Per-fabric adaptive thresholds did not help** with this little data per fabric (see the ablation).

## Files

- `Group13_AFID_results.xlsx`, with these sheets:
  - `Summary`, `Protocol`
  - `Image_Level`: all variants, both threshold modes, all, fair and capture-shift subsets
  - `Pixel_Level`: raw, post and Hough masks; pooled and per-image
  - `Window_Level`, `Ablation_Fixed_vs_Adaptive`
  - `Per_Fabric`, `Per_Defect`, `Per_Image` (all 245 strips)
  - `Confidence`, `Model_Selection`, `Thresholds` (per fold)
- `figures/`:
  - ROC, confusion matrices (V3 and selected, all and fair fabrics), score histogram
  - ablation, pixel-metric, per-fabric and per-defect bar charts
  - best, median and worst example crops
- `masks/`: the predicted mask of every strip (245).
- `overlays/`: prediction in red, ground-truth contour in green, with a banner for score, P(defect), reliability and IoU. These, and `deviation_maps/` (window × pixel probability, inferno colour map), cover every defective strip, every false positive and 10 true negatives (134 strips).

Reproduce with `python Codes/run_all.py`. Feature extraction takes about 8 min and the nested CV about 70 min, on 3 worker processes.
