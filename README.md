# DIP_2026_Project17_Group13: Adaptive Fabric Surface Defect Detection and Localization using Classical Texture Descriptors

ECE501 Digital Image Processing, Group 13 (Prakruti Patel, Kushal Rajyaguru, Chintan Thacker,
Parthiv Karangiya). AITEX Fabric Image Database (AFID), classical methods only, no deep learning.

| Folder | Contents |
|---|---|
| `Codes/` | All code, `requirements.txt`, one-command pipeline (`Codes/run_all.py`), see `Codes/README.md` |
| `Results/` | Excel workbook, figures (ROC, confusion matrices, metric bars), overlays, deviation maps, masks; protocol and headline numbers in `Results/README.md` |
| `Mid_Sem_Report/` | Mid-semester report and presentation |
| `data/` | AFID (not committed): `NODefect_images/`, `Defect_images/`, `Mask_images/`; `Codes/download_afid.py` fetches it |

```bash
python3 -m venv .venv && .venv/bin/pip install -r Codes/requirements.txt
.venv/bin/python Codes/download_afid.py
.venv/bin/python Codes/run_all.py
```
