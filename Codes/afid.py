"""AFID (AITEX Fabric Image Database) listing and ground-truth masks.

Filenames are nnnn_ddd_ff.png: image id, defect code (000 = defect-free), fabric 00-06.
Known quirks handled here:
  * NODefect_images/2608691-202020u/0018_00_01.png has a malformed name -> skipped by the regex.
  * 0100_025_08.png is fabric 08 with no mask -> skipped.
  * 0044_019_04 and 0097_030_03 have their mask split into _mask1 + _mask2 -> merged (OR).
  * 0106_010_03 has an empty mask -> kept (counts as defective, image-level only).
  * 0094_027_05.png is 3796 px wide -> handled by every stage (no fixed width).
"""

import re
from pathlib import Path

import cv2
import numpy as np

NAME = re.compile(r"^(\d{4})_(\d{3})_(\d{2})$")

DEFECT_NAMES = {
    "000": "no defect", "002": "broken end", "006": "broken yarn", "010": "broken pick",
    "016": "weft curling", "019": "fuzzy ball", "022": "cut selvage", "023": "crease",
    "025": "warp ball", "027": "knot", "029": "contamination", "030": "nep", "036": "weft crack",
}


def list_images(root):
    """Return a list of dicts (name, path, masks, label, defect, fabric) for every usable strip."""
    root = Path(root)
    masks = {}
    for m in (root / "Mask_images").glob("*.png"):
        stem = re.sub(r"_mask\d?$", "", m.stem)
        masks.setdefault(stem, []).append(m)
    out = []
    for p in sorted(root.glob("NODefect_images/**/*.png")) + sorted(root.glob("Defect_images/*.png")):
        g = NAME.match(p.stem)
        if not g:
            continue
        _, code, fabric = g.groups()
        defective = code != "000"
        if defective and p.stem not in masks:
            continue
        if fabric not in {f"{i:02d}" for i in range(7)}:
            continue
        out.append({
            "name": p.stem, "path": str(p), "masks": [str(m) for m in sorted(masks.get(p.stem, []))],
            "label": int(defective), "defect": code, "fabric": fabric,
        })
    return out


def read_image(rec):
    return cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)


def read_mask(rec, shape):
    """Binary ground truth (uint8 0/1); all-zero for defect-free strips; split masks are merged."""
    gt = np.zeros(shape, np.uint8)
    for m in rec["masks"]:
        a = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if a.shape != shape:
            a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        gt |= (a > 127).astype(np.uint8)
    return gt


if __name__ == "__main__":
    import sys
    recs = list_images(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "data")
    n_def = sum(r["label"] for r in recs)
    print(f"{len(recs) - n_def} defect-free + {n_def} defective = {len(recs)}")
    assert (len(recs) - n_def, n_def) == (140, 105), "expected 140 + 105 usable AFID strips"
    assert sum(len(r["masks"]) == 2 for r in recs) == 2, "expected two split masks"
