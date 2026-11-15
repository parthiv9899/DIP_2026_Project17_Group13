"""Aggregates the outer-fold test predictions into the Excel workbook, figures and image outputs.

Ratios with an empty denominator are NaN (never 0). Pixel metrics are given pooled (TP/FP/FN summed
over images) and as per-image means; tolerance metrics count a predicted pixel as correct if a GT
pixel lies within d px (precision) and a GT pixel as found if a prediction lies within d px (recall).
"""

import pickle
import shutil
from itertools import product
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import afid
from pipeline import FINAL, VARIANTS

SEL = "SEL_inner_cv_selected"  # per outer fold, the variant with the best inner out-of-fold accuracy
VARS = [*VARIANTS, SEL]
FAIR = {"03", "04", "06"}  # fabrics without the defect/defect-free capture shift
SHIFTED = {"00", "01", "02", "05"}


def ratio(a, b):
    return a / b if b > 0 else np.nan


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else np.nan


def pixel_pooled(cs):
    tp, fp, fn = (sum(c[k] for c in cs) for k in ("tp", "fp", "fn"))
    out = {"pixel_precision": ratio(tp, tp + fp), "pixel_recall": ratio(tp, tp + fn)}
    out["pixel_f1"] = f1(out["pixel_precision"], out["pixel_recall"])
    out["pixel_iou"] = ratio(tp, tp + fp + fn)
    for d in (2, 3):
        p = ratio(sum(c[f"tolP{d}"] for c in cs), sum(c["npred"] for c in cs))
        r = ratio(sum(c[f"tolR{d}"] for c in cs), sum(c["ngt"] for c in cs))
        out.update({f"tol{d}_precision": p, f"tol{d}_recall": r, f"tol{d}_f1": f1(p, r)})
    return out


def pixel_one(c):
    """Per-image pixel metrics; precision is NaN with no prediction, recall/IoU NaN with no GT."""
    p, r = ratio(c["tp"], c["tp"] + c["fp"]), ratio(c["tp"], c["tp"] + c["fn"])
    out = {"pixel_precision": p, "pixel_recall": r,
           "pixel_f1": f1(p, r) if not (np.isnan(p) or np.isnan(r)) else (0.0 if c["ngt"] and c["npred"] == 0 else np.nan),
           "pixel_iou": ratio(c["tp"], c["tp"] + c["fp"] + c["fn"])}
    for d in (2, 3):
        tp_, tr_ = ratio(c[f"tolP{d}"], c["npred"]), ratio(c[f"tolR{d}"], c["ngt"])
        out[f"tol{d}_f1"] = f1(tp_, tr_) if not (np.isnan(tp_) or np.isnan(tr_)) else (0.0 if c["ngt"] else np.nan)
    return out


def image_metrics(y, pred, score):
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    p, r, spec = ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {"n": len(y), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": ratio(tp + tn, len(y)),
            "precision": p, "recall": r, "specificity": spec, "f1": f1(p, r),
            "balanced_accuracy": (r + spec) / 2, "image_iou": ratio(tp, tp + fp + fn),
            "roc_auc": roc_auc_score(y, score) if len(set(y)) == 2 else np.nan}


def select(d):
    """Variant with the best inner out-of-fold image accuracy (ties: AUC); test labels are never used."""
    names = list(d["oof_scores"])
    y = np.array([int(n.split("_")[1] != "000") for n in names])
    out = []
    for v in VARIANTS:
        s = np.array([d["oof_scores"][n][v] for n in names])
        out.append({"variant": v, "inner_oof_accuracy": ((s >= d["thresholds"][v]["img"]) == y).mean(),
                    "inner_oof_auc": roc_auc_score(y, s)})
    best = max(out, key=lambda o: (o["inner_oof_accuracy"], o["inner_oof_auc"]))["variant"]
    return best, out


def load(work):
    rows, thr, sel = [], [], []
    for p in sorted(work.glob("fold*.pkl")):
        d = pickle.load(open(p, "rb"))
        k = int(p.stem[4:])
        best, table = select(d)
        sel += [{"fold": k, **t, "selected": t["variant"] == best} for t in table]
        for r in d["rows"]:
            r["variants"][SEL] = r["variants"][best] | {"chosen": best}
        rows += d["rows"]
        for v, t in d["thresholds"].items():
            thr.append({"fold": k, "variant": v, **{x: t[x] for x in ("img", "tw", "tp", "win", "inner_pixel_f1")},
                        "per_fabric": "; ".join(f"{f}: img {ft['img']:.3f} tw {ft['tw']} tp {ft['tp']}"
                                                 for f, ft in t["fabric"].items())})
    return rows, pd.DataFrame(thr), pd.DataFrame(sel)


# --- Tables ----------------------------------------------------------------------------------


def per_image(rows, mode="global"):
    out = []
    for r in rows:
        v = r["variants"][FINAL]
        m = v[mode]
        rec = {"image": r["name"], "fold": r["fold"], "fabric": r["fabric"], "defect_code": r["defect"],
               "defect": afid.DEFECT_NAMES[r["defect"]], "label": r["label"], "score": v["score"],
               "confidence_p_defect": v["prob"], "pred": m["pred"], "correct": int(m["pred"] == r["label"]),
               "conformal_p_normal": v["p_normal"], "conformal_p_defect": v["p_defect"],
               "prediction_set": v["set"], "reliability": v["reliability"],
               "gt_px": m["post"]["ngt"], "pred_px": m["post"]["npred"], "TP_px": m["post"]["tp"],
               "FP_px": m["post"]["fp"], "FN_px": m["post"]["fn"]}
        sv = r["variants"][SEL]
        rec.update({"selected_variant": sv["chosen"], "selected_score": sv["score"],
                    "selected_pred": sv["global"]["pred"]})
        rec.update(pixel_one(m["post"]))
        rec["pixel_iou_hough"] = pixel_one(m["hough"])["pixel_iou"]
        rec["pixel_f1_hough"] = pixel_one(m["hough"])["pixel_f1"]
        out.append(rec)
    return pd.DataFrame(out)


def image_level(rows):
    out = []
    for v in VARS:
        for mode in ("global", "fabric"):
            for subset, keep in (("all fabrics", None), ("fair: fabrics 03/04/06", FAIR),
                                 ("capture-shift: 00/01/02/05", SHIFTED)):
                rs = [r for r in rows if keep is None or r["fabric"] in keep]
                m = image_metrics([r["label"] for r in rs], [r["variants"][v][mode]["pred"] for r in rs],
                                  [r["variants"][v]["score"] for r in rs])
                out.append({"variant": v, "thresholds": mode, "subset": subset, **m})
    return pd.DataFrame(out)


def pixel_level(rows):
    out = []
    for mode in ("global", "fabric"):
        for mask in ("raw", "post", "hough"):
            cs = [r["variants"][FINAL][mode][mask] for r in rows]
            cd = [c for c, r in zip(cs, rows) if r["label"]]
            per = pd.DataFrame([pixel_one(c) for c in cd])
            for scope, val in (("pooled, defective images", pixel_pooled(cd)),
                               ("pooled, all test images", pixel_pooled(cs)),
                               ("per-image mean, defective images", per.mean().to_dict())):
                out.append({"thresholds": mode, "mask": mask, "scope": scope, **val})
    return pd.DataFrame(out)


def window_level(rows):
    out = []
    for v in VARS:
        tp, fp, fn = np.sum([r["variants"][v]["win"] for r in rows], 0)
        p, rc = ratio(tp, tp + fp), ratio(tp, tp + fn)
        out.append({"variant": v, "cells_TP": tp, "cells_FP": fp, "cells_FN": fn, "precision": p, "recall": rc,
                    "f1": f1(p, rc), "iou": ratio(tp, tp + fp + fn)})
    return pd.DataFrame(out)


def ablation(rows, img, win):
    out = []
    for v in VARS:
        for mode in ("global", "fabric"):
            a = img[(img.variant == v) & (img.thresholds == mode)].set_index("subset")
            cd = [r["variants"][v][mode]["raw"] for r in rows if r["label"]]
            pp = pixel_pooled(cd)
            out.append({"variant": v, "thresholds": mode,
                        "image_accuracy": a.loc["all fabrics", "accuracy"],
                        "image_f1": a.loc["all fabrics", "f1"], "image_auc": a.loc["all fabrics", "roc_auc"],
                        "fair_accuracy_03_04_06": a.loc["fair: fabrics 03/04/06", "accuracy"],
                        "fair_auc_03_04_06": a.loc["fair: fabrics 03/04/06", "roc_auc"],
                        "window_f1": win.set_index("variant").loc[v, "f1"],
                        "pixel_f1_pooled": pp["pixel_f1"], "pixel_iou_pooled": pp["pixel_iou"],
                        "tol3_f1_pooled": pp["tol3_f1"]})
    return pd.DataFrame(out)


def grouped(rows, df, key, labels):
    out = []
    for g, part in df.groupby(key):
        rs = [r for r in rows if r["name"] in set(part.image)]
        cd = [r["variants"][FINAL]["global"]["post"] for r in rs if r["label"]]
        m = image_metrics(part.label, part.pred, part.score)
        d = part[part.label == 1]
        rec = {key: g, "name": labels(g), "n_images": len(part), "n_defective": int(part.label.sum()),
               "image_accuracy": m["accuracy"], "image_recall": m["recall"],
               "image_specificity": m["specificity"], "image_f1": m["f1"]}
        if cd:
            rec.update({f"pooled_{k}": v for k, v in pixel_pooled(cd).items()
                        if k in ("pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "tol3_f1")})
            rec["mean_pixel_iou"] = d.pixel_iou.mean()
            rec["mean_tol3_f1"] = d.tol3_f1.mean()
        out.append(rec)
    return pd.DataFrame(out)


def confidence(df):
    out = []
    for rel, part in df.groupby("reliability"):
        out.append({"table": "reliability", "group": rel, "n": len(part), "accuracy": part.correct.mean(),
                    "share": len(part) / len(df)})
    bins = np.linspace(0, 1, 6)
    for lo, hi in zip(bins[:-1], bins[1:]):
        part = df[(df.confidence_p_defect >= lo) & (df.confidence_p_defect < hi + (hi == 1))]
        out.append({"table": "calibration", "group": f"P(defect) {lo:.1f}-{hi:.1f}", "n": len(part),
                    "observed_defect_rate": part.label.mean() if len(part) else np.nan,
                    "mean_confidence": part.confidence_p_defect.mean() if len(part) else np.nan})
    return pd.DataFrame(out)


# --- Figures ---------------------------------------------------------------------------------


def figures(rows, df, img, abl, perf, perd, out):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in VARS:
        y = [r["label"] for r in rows]
        s = [r["variants"][v]["score"] for r in rows]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2 if v == FINAL else 1, label=f"{v} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Image-level ROC (outer-fold test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_image_level.png", dpi=150)
    plt.close(fig)

    for (model, col), (subset, keep) in product(((FINAL, "pred"), ("selected", "selected_pred")),
                                                (("all", None), ("fair_03_04_06", FAIR))):
        d = df if keep is None else df[df.fabric.isin(keep)]
        cm = pd.crosstab(d.label, d[col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, cmap="Blues")
        for (i, j), n in np.ndenumerate(cm):
            ax.text(j, i, n, ha="center", va="center", fontsize=14, color="white" if n > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "defect"], yticklabels=["normal", "defect"],
               xlabel="Predicted", ylabel="Actual",
               title=f"{model}, {subset}: acc {(cm[0, 0] + cm[1, 1]) / cm.sum():.3f}")
        fig.tight_layout()
        fig.savefig(out / f"confusion_matrix_{model}_{subset}.png", dpi=150)
        plt.close(fig)

    a = abl[abl.thresholds == "global"].set_index("variant")
    cols = ["image_accuracy", "fair_accuracy_03_04_06", "image_auc", "window_f1", "pixel_f1_pooled", "tol3_f1_pooled"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(a)
    for i, v in enumerate(a.index):
        ax.bar(np.arange(len(cols)) + i * w, a.loc[v, cols].astype(float), w, label=v)
    ax.set_xticks(np.arange(len(cols)) + 0.4 - w / 2, [c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set(ylim=(0, 1), title="Fixed vs adaptive texture parameters (global thresholds)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "ablation_fixed_vs_adaptive.png", dpi=150)
    plt.close(fig)

    for name, t, key in (("per_fabric", perf, "fabric"), ("per_defect", perd, "name")):
        cols = [c for c in ("image_accuracy", "image_recall", "pooled_pixel_f1", "pooled_pixel_iou", "pooled_tol3_f1")
                if c in t]
        fig, ax = plt.subplots(figsize=(max(7, len(t) * 0.9), 4.5))
        w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(np.arange(len(t)) + i * w, t[c].astype(float), w, label=c)
        ax.set_xticks(np.arange(len(t)) + 0.4 - w / 2, t[key].astype(str), rotation=30, ha="right", fontsize=8)
        ax.set(ylim=(0, 1.05), title=name.replace("_", " ").title() + f" metrics ({FINAL})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{name}_metrics.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, name in ((0, "defect-free"), (1, "defective")):
        ax.hist(df[df.label == lab].score, bins=30, range=(0, 1), alpha=0.6, label=name)
    ax.set(xlabel="Image score (top-3 window probability)", ylabel="Strips", title="Score distribution (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "score_histogram.png", dpi=150)
    plt.close(fig)

    pix = pd.DataFrame([pixel_pooled([r["variants"][FINAL]["global"][m] for r in rows if r["label"]]) | {"mask": m}
                        for m in ("raw", "post", "hough")]).set_index("mask")
    cols = ["pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "tol2_f1", "tol3_f1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(pix.index):
        ax.bar(np.arange(len(cols)) + i * 0.27, pix.loc[m, cols].astype(float), 0.27, label=m)
    ax.set_xticks(np.arange(len(cols)) + 0.27, cols, fontsize=8)
    ax.set(ylim=(0, 1), title="Pixel-level metrics, pooled over defective test strips")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pixel_metrics.png", dpi=150)
    plt.close(fig)


def images(recs, df, maps, out):
    """Masks for every strip; overlays and deviation maps for defective, false-positive and 10 normal strips."""
    by = {r["name"]: r for r in recs}
    for d in ("masks", "overlays", "deviation_maps"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tn = df[(df.label == 0) & (df.pred == 0)].image.tolist()[:10]
    show = set(df[(df.label == 1) | (df.pred == 1)].image) | set(tn)
    for _, r in df.iterrows():
        n = r.image
        shutil.copy(maps / f"{n}_mask.png", out / "masks" / f"{n}_pred_mask.png")
        if n not in show:
            continue
        img = afid.read_image(by[n])
        gt = afid.read_mask(by[n], img.shape)
        pred = cv2.imread(str(maps / f"{n}_mask.png"), 0) > 0
        dev = cv2.imread(str(maps / f"{n}_deviation.png"), 0)
        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ov[pred] = (0.5 * ov[pred] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, (0, 255, 0), 1)
        text = (f"{n} {r.defect} | score {r.score:.2f} P(defect) {r.confidence_p_defect:.2f} | "
                f"{r.reliability} | IoU {r.pixel_iou:.2f}")
        banner = np.full((28, ov.shape[1], 3), 255, np.uint8)
        cv2.putText(banner, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "overlays" / f"{n}_overlay.jpg"), np.vstack([banner, ov]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(out / "deviation_maps" / f"{n}_deviation.jpg"),
                    cv2.applyColorMap(dev, cv2.COLORMAP_INFERNO), [cv2.IMWRITE_JPEG_QUALITY, 88])


def examples(recs, df, maps, out):
    """Crops (1024 px) around the defect: original, ground truth, deviation map, prediction."""
    by = {r["name"]: r for r in recs}
    d = df[df.label == 1].sort_values("pixel_iou", ascending=False)
    pick = pd.concat([d.head(4), d.iloc[len(d) // 2 - 1:len(d) // 2 + 1], d.dropna(subset=["pixel_iou"]).tail(2)]).drop_duplicates("image")
    fig, axes = plt.subplots(len(pick), 4, figsize=(16, 1.25 * len(pick) + 0.6))
    for row, (_, r) in zip(axes, pick.iterrows()):
        img = afid.read_image(by[r.image])
        gt = afid.read_mask(by[r.image], img.shape)
        xs = np.flatnonzero(gt.any(0))
        c = int(xs.mean()) if len(xs) else img.shape[1] // 2
        x0 = int(np.clip(c - 512, 0, img.shape[1] - 1024))
        sl = np.s_[:, x0:x0 + 1024]
        dev = cv2.imread(str(maps / f"{r.image}_deviation.png"), 0)
        pred = cv2.imread(str(maps / f"{r.image}_mask.png"), 0)
        for ax, im, t, cm in zip(row, (img, gt, dev, pred), ("image", "ground truth", "deviation map", "prediction"),
                                 ("gray", "gray", "inferno", "gray")):
            ax.imshow(im[sl], cmap=cm, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{r.image} ({r.defect}) {t}" + (f", IoU {r.pixel_iou:.2f}" if t == "prediction" else ""),
                         fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "examples_best_median_worst.png", dpi=130)
    plt.close(fig)


# --- Workbook -------------------------------------------------------------------------------


def build(recs, work, out):
    rows, thr, sel = load(work)
    assert len(rows) == len(recs), f"{len(rows)} test rows for {len(recs)} strips: a fold is missing"
    out.mkdir(parents=True, exist_ok=True)
    df = per_image(rows)
    img = image_level(rows)
    pix = pixel_level(rows)
    win = window_level(rows)
    abl = ablation(rows, img, win)
    perf = grouped(rows, df, "fabric", lambda f: f"fabric {f}")
    perd = grouped(rows, df, "defect_code", lambda c: f"{c} {afid.DEFECT_NAMES[c]}")
    conf = confidence(df)
    fi = img[(img.variant == FINAL) & (img.thresholds == "global")].set_index("subset")
    pp = pix[(pix.thresholds == "global") & (pix["mask"] == "post")].set_index("scope")
    summary = pd.DataFrame([
        (f"Image accuracy, all {len(rows)} strips", fi.loc["all fabrics", "accuracy"]),
        ("Image precision / recall / F1", f"{fi.loc['all fabrics', 'precision']:.3f} / "
                                          f"{fi.loc['all fabrics', 'recall']:.3f} / {fi.loc['all fabrics', 'f1']:.3f}"),
        ("Image ROC AUC", fi.loc["all fabrics", "roc_auc"]),
        ("Image accuracy, fair fabrics 03/04/06", fi.loc["fair: fabrics 03/04/06", "accuracy"]),
        ("Image accuracy, capture-shift fabrics 00/01/02/05", fi.loc["capture-shift: 00/01/02/05", "accuracy"]),
        *[(f"Inner-CV selected model: image {k}, {s}", img[(img.variant == SEL) & (img.thresholds == "global")]
           .set_index("subset").loc[s, k]) for s in ("all fabrics", "fair: fabrics 03/04/06")
          for k in ("accuracy", "precision", "recall", "f1", "roc_auc")],
        ("Inner-CV selected variant per fold", ", ".join(
            f"{r.fold}: {r.variant}" for r in sel[sel.selected].itertuples())),
        ("Window (16-px cell) F1", win.set_index("variant").loc[FINAL, "f1"]),
        *[(f"Pixel {k} ({s})", pp.loc[s, k]) for s in pp.index
          for k in ("pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "tol2_f1", "tol3_f1")],
        ("Reliable (conformal singleton) share", conf[conf.group == "reliable"].share.sum()),
        ("Accuracy on reliable strips", conf[conf.group == "reliable"].accuracy.sum()),
    ], columns=["metric (final model " + FINAL + ", global thresholds)", "value"])
    protocol = pd.DataFrame({"protocol": [
        "Data: AFID, 140 defect-free + 105 defective strips (256 x 4096), masks merged where split.",
        "Outer 5-fold CV stratified by (label, fabric), split by image: every strip tested once.",
        "Inner 3-fold CV on each outer training set chooses all thresholds, the calibrator and conformal references.",
        "No threshold or hyperparameter was chosen on test images. Undefined ratios are NaN.",
        "Window model: gradient-boosted trees on LBP/GLCM/Gabor/filter-bank window descriptors.",
        "Pixel model: gradient-boosted trees on the per-pixel filter bank; mask = window prob >= tw AND pixel prob >= tp, "
        "gated by the image decision, components < 10 px dropped ('post'); 'hough' adds Hough line linking.",
        "Variants V0-V4 change one thing at a time from fixed single-scale parameters to adaptive multi-scale ones.",
        "'fabric' thresholds are per-fabric (adaptive) where the inner set has >= 3 defective strips of that fabric.",
        "Fabrics 00/01/02/05 show a global capture shift between defective and defect-free strips; 03/04/06 are the fair test.",
    ]})
    with pd.ExcelWriter(out / "Group13_AFID_results.xlsx", engine="openpyxl") as xw:
        for name, t in (("Summary", summary), ("Image_Level", img), ("Pixel_Level", pix), ("Window_Level", win),
                        ("Ablation_Fixed_vs_Adaptive", abl), ("Per_Fabric", perf), ("Per_Defect", perd),
                        ("Per_Image", df), ("Confidence", conf), ("Model_Selection", sel), ("Thresholds", thr), ("Protocol", protocol)):
            t.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    60, max(10, max(len(str(c.value)) for c in col[:50]) + 2))
    figures(rows, df, img, abl, perf, perd, out / "figures")
    images(recs, df, work / "maps", out)
    examples(recs, df, work / "maps", out / "figures")
    print(summary.to_string(index=False))
    return summary
