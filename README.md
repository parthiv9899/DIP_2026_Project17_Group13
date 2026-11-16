# Adaptive Fabric Surface Defect Detection and Localization

### Classical Computer Vision | Texture Analysis | Machine Learning | Image Segmentation

**Group 13 · Project 17**  
School of Engineering and Applied Science  
Ahmedabad University, Ahmedabad, India

---

## Overview

Fabric manufacturing is highly sensitive to surface imperfections. Defects such as broken picks, broken yarns, fuzzy balls and broken ends can reduce the quality and usability of the final textile product.

Automatically detecting these imperfections is challenging because fabrics naturally contain repetitive patterns and variations in texture. A small irregularity may indicate an actual defect, but it may also be part of the normal weave structure.

This project develops a **fabric surface defect detection and localization framework using classical image processing, handcrafted texture descriptors and machine learning techniques**.

The objective is not only to determine whether a fabric image contains a defect, but also to identify its approximate location and generate a pixel-level prediction mask.

> By characterizing normal fabric texture and identifying local deviations from it, the system aims to detect and localize surface defects.

## Project Objectives

- Detect whether a fabric image contains a surface defect.
- Localize defects using pixel-level prediction masks.
- Extract texture information using classical descriptors such as GLCM and LBP.
- Investigate local intensity and texture deviations for anomaly detection.
- Evaluate the system using image-level and pixel-level metrics.
- Analyze false positives and limitations in defect localization.

## Dataset

Experiments use the **AITEX Fabric Image Database (AFID)**, which provides grayscale fabric images and corresponding ground-truth defect masks.

| Dataset property | Configuration |
|---|---:|
| Dataset | AITEX AFID |
| Training images | 84 |
| Testing images | 21 |
| Defective images available | 106 |
| Ground-truth masks available | 107 |
| Patch size in evaluated baseline | 16 × 16 pixels |
| Threshold selection | 98th percentile |
| Evaluation | Pixel-level comparison |

Training and testing were separated at image level. Ground-truth masks were used for quantitative evaluation, and the held-out test set was kept separate from training.

## Methodology

The framework combines classical texture analysis with local anomaly detection.

### 1. Image preparation

Fabric images and their corresponding ground-truth masks are organized and matched before feature extraction and evaluation.

### 2. Texture analysis

**Gray-Level Co-occurrence Matrix (GLCM)**

GLCM measures spatial relationships between pixel intensities. The extracted statistics include:

- Contrast
- Homogeneity
- Energy
- Correlation

These features help characterize differences in fabric texture.

**Local Binary Patterns (LBP)**

LBP captures local texture patterns by comparing the intensity of a pixel with its neighbours. Histogram-based descriptors represent micro-level variations in the fabric surface.

**Local dark contrast**

Local intensity differences are used to identify unusually dark regions. This method forms part of the evaluated AITEX baseline detector.

### 3. Patch-based analysis and defect localization

The evaluated baseline uses 16 × 16 patches to analyze local fabric regions. A threshold based on the 98th percentile is used to identify candidate anomalous regions.

The resulting predictions are represented as binary masks and compared with the corresponding ground-truth masks.

### 4. Evaluation and error analysis

The system is evaluated using:

- Precision
- Recall
- F1-score
- Intersection over Union (IoU)

Visual comparisons between the original fabric image, ground-truth mask, deviation map and predicted mask are also used to inspect localization quality and identify false positives.

## Experimental Results

The following quantitative results are from the held-out test-set evaluation presented in the project presentation.

| Metric | Result |
|---|---:|
| Image-level classification accuracy | **80.4%** |
| Image-level ROC-AUC | **87.6%** |
| Pixel-level localization F1-score | **36.2%** |
| Pixel-level F1-score with ±3-pixel tolerance | **45.6%** |

The results indicate that image-level classification is stronger than precise pixel-level localization. Allowing a tolerance of three pixels improves the localization F1-score, reflecting the sensitivity of thin defects to small boundary shifts.

## Qualitative Results

The examples below show the stronger localization cases from the generated visual evaluation. Lower-performing examples have been omitted to keep this section focused on representative successful predictions.

![Selected fabric defect localization examples: original image, ground truth, deviation map and prediction](Results/figures/qualitative_results.png)

Each row contains:

1. Original fabric image.
2. Ground-truth defect mask.
3. Deviation map.
4. Predicted defect mask and its IoU.

| Defect example | Prediction IoU |
|---|---:|
| Broken pick | **0.70** |
| Broken pick | **0.52** |
| Broken yarn | **0.49** |
| Fuzzy ball | **0.44** |

These examples illustrate successful localization of both elongated defects and a small localized defect. They are selected qualitative examples and should not be interpreted as the average performance across the test set.

## Challenges and Observations

The main challenges identified during evaluation are:

- **Small defects:** Fuzzy balls, neps and warp balls can be difficult to distinguish from normal fabric texture.
- **Pixel-level localization:** Thin defects are sensitive to small boundary shifts during prediction.
- **Fabric variations:** Different fabric structures and image-capture conditions affect detection performance.
- **Threshold selection:** Per-fabric thresholds can overfit when only a few training examples are available.
- **False positives:** Normal texture variations can sometimes be incorrectly identified as defects.

The system performs better on thread-related defects, while small localized defects remain challenging due to their similarity to normal fabric variations.

## Future Scope

Potential improvements include:

1. Optimizing adaptive threshold selection.
2. Reducing false-positive regions.
3. Investigating smaller patch sizes.
4. Improving defect boundary localization.
5. Exploring GLCM-LBP feature fusion with local contrast.
6. Evaluating improvements on the same held-out test set.

## Technology Stack

| Technology | Application |
|---|---|
| Python | Core implementation |
| OpenCV | Image processing and segmentation |
| NumPy | Numerical operations |
| SciPy | Signal processing |
| scikit-image | Texture feature extraction |
| scikit-learn | Machine learning and evaluation |
| Pandas | Results processing |
| Matplotlib | Visualization |
| OpenPyXL | Excel report generation |

**Approach:** Classical image processing and machine learning.

## Repository Structure

```text
Fabric-Defect-Detection/
│
├── Codes/
│   ├── afid.py
│   ├── download_afid.py
│   ├── features.py
│   ├── pipeline.py
│   ├── report.py
│   └── run_all.py
│
├── Images/
│   ├── Defect_images/
│   ├── Mask_images/
│   └── NODefect_images/
│
├── Mid_Sem_Report/
│
├── Results/
│   ├── deviation_maps/
│   ├── figures/
│   ├── masks/
│   ├── overlays/
│   └── Group13_AFID_results.xlsx
│
├── README.md
└── .gitignore
```

The dataset and generated outputs may be large. Follow the repository's `.gitignore` rules when committing files.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r Codes/requirements.txt
```

### 4. Prepare the dataset

Place the AFID image and mask folders in the locations expected by the code. If using the provided download script, run:

```bash
python Codes/download_afid.py
```

### 5. Run the pipeline

```bash
python Codes/run_all.py
```

The pipeline performs feature extraction, cross-validation and report generation. Refer to the individual scripts for available options and expected data paths.

## Project Team

**Group 13 · Project 17**

| Team Member | Enrollment Number |
|---|---|
| Prakruti Patel | AU2440046 |
| Kushal Rajyaguru | AU2440140 |
| Chintan Thacker | AU2440215 |
| Parthiv Karangiya | AU2440016 |

**School of Engineering and Applied Science**  
Ahmedabad University, Gujarat, India

---

## References

1. F. Tajeripour, E. Kabir, and A. Sheikhi, “Fabric Defect Detection Using Modified Local Binary Patterns,” *Journal on Advances in Signal Processing*, 2007.
2. X. Zhang and X. Fan, “Fabric Defect Detection based on GLCM Approach,” ICIMMS, 2016.
3. L. Zhang, J. Jing, and H. Zhang, “Fabric Defect Classification Based on LBP and GLCM,” *Journal of Fiber Bioengineering and Informatics*, vol. 8, no. 1, pp. 81–89, 2015.
4. A. S. Tolba et al., “Decision Fusion for Visual Inspection of Textiles,” *Textile Research Journal*, 2010.
5. “Fabric Defect Detection based on GLCM and Gabor Filter: A Comparison,” *Optik*, 2013.
