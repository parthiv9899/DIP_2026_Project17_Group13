"""Group 13 adaptive texture-deviation pipeline: feature cache, nested cross-validation, test outputs.

Protocol (all splits are by image):
  outer 5-fold stratified by (label, fabric): every strip is tested exactly once, by models that
  never saw it. Inside each outer training set a 3-fold inner CV produces out-of-fold scores that
  choose every threshold (image, window, pixel), the isotonic calibrator and the conformal
  reference scores. The test fold is touched only once, with everything frozen.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import pickle
import zlib
import time
from multiprocessing import Pool, Process
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import afid
import features as F

cv2.setNumThreads(1)
SEED = 13
WORKERS = 3  # this Mac is shared with another long job
POS_MIN = 8  # a cell is defective with >= 8 GT pixels, normal with 0, ignored in between
TW = np.round(np.arange(0.1, 0.91, 0.1), 2)  # window-probability thresholds searched
TP = np.round(np.arange(0.05, 0.96, 0.05), 2)  # pixel-probability thresholds searched
MIN_AREA = 10  # px; smaller predicted components are dropped
ALPHA = 0.1  # conformal miscoverage for the reliability label
FINAL = "V3_adaptive"
F16_MAX = 65_000

GROUPS_FIXED = {"int", "lbp", "glcm1", "gabF", "chan"}
GROUPS_ADAPT = GROUPS_FIXED | {"glcmA", "gabA"}
# Fixed-vs-adaptive ladder: one change per step.
VARIANTS = {
    "V0_fixed": ({32}, GROUPS_FIXED, ("raw",)),
    "V1_multiscale": ({32, 64}, GROUPS_FIXED, ("raw",)),
    "V2_ms_adaptive_params": ({32, 64}, GROUPS_ADAPT, ("raw",)),
    "V3_adaptive": ({32, 64}, GROUPS_ADAPT, ("z",)),
    "V4_adaptive_raw+z": ({32, 64}, GROUPS_ADAPT, ("raw", "z")),
}


def columns(names, variant):
    scales, groups, norms = VARIANTS[variant]
    out = []
    for i, n in enumerate(names):
        norm = "z" if n.startswith("z|") else "raw"
        s, g, _ = n.removeprefix("z|").split(":")
        if int(s[1:]) in scales and g in groups and norm in norms:
            out.append(i)
    return np.array(out)


# --- Feature cache -------------------------------------------------------------------------


def _extract_one(args):
    rec, cache = args
    out = cache / f"{rec['name']}.npz"
    if out.exists():
        return rec["name"]
    img = afid.read_image(rec)
    gt = afid.read_mask(rec, img.shape)
    a = F.analyse(img)
    xp, yp = F.sample_pixels(a["ch"], gt, rec["name"], a["fabric"])
    np.savez(out, X=f16(a["X"]), counts=F.cell_counts(gt).astype(np.int16),
             Xp=xp, yp=yp, px=a["px"], py=a["py"], shape=np.array(img.shape), fabric=a["fabric"])
    names = cache / "feature_names.json"
    if not names.exists():
        names.write_text(json.dumps(a["names"]))
    return rec["name"]


def f16(X):
    """float16 for the cache; saturates the few raw long-wavelength Gabor energies above its range."""
    return np.clip(np.nan_to_num(X, posinf=F16_MAX, neginf=-F16_MAX), -F16_MAX, F16_MAX).astype(np.float16)


def extract(recs, cache):
    cache.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=20) as pool:
        for i, n in enumerate(pool.imap_unordered(_extract_one, [(r, cache) for r in recs]), 1):
            if i % 25 == 0:
                print(f"  features {i}/{len(recs)}", flush=True)


# --- Models ----------------------------------------------------------------------------------


def gbt(max_iter):
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          early_stopping=False, random_state=SEED)


def load_pool(recs, cache):
    """Per image: labelled training cells (all defective, a fixed 12% of normal) and pixel samples."""
    pool = {}
    for r in recs:
        z = np.load(cache / f"{r['name']}.npz")
        c = z["counts"]
        rng = np.random.default_rng(zlib.crc32(r["name"].encode()))
        keep = ((c >= POS_MIN) | ((c == 0) & (rng.random(len(c)) < 0.12))) & F.cell_roi(z["fabric"], z["shape"])
        pool[r["name"]] = (f16(z["X"][keep].astype(np.float32)), (c[keep] >= POS_MIN).astype(np.uint8), z["Xp"], z["yp"])
    return pool


def fit(pool, names, colsets):
    """Window model per variant and one pixel model, all trained on the given images only."""
    Xw = np.concatenate([pool[n][0] for n in names])
    yw = np.concatenate([pool[n][1] for n in names])
    models = {v: gbt(200).fit(Xw[:, c].astype(np.float32), yw) for v, c in colsets.items()}
    Xp = np.concatenate([pool[n][2] for n in names]).astype(np.float32)
    yp = np.concatenate([pool[n][3] for n in names])
    models["pixel"] = gbt(150).fit(Xp, yp)
    return models


def upsample(cells, shape):
    """Cell probabilities (rows, cols) -> per-pixel map by bilinear interpolation, edge-padded."""
    h, w = cells.shape
    up = cv2.resize(cells.astype(np.float32), (w * F.CELL, h * F.CELL), interpolation=cv2.INTER_LINEAR)
    return np.pad(up, ((0, shape[0] - up.shape[0]), (0, shape[1] - up.shape[1])), mode="edge")


def predict(models, colsets, rec, cache):
    """Per variant cell probabilities and image score; pixel probability map (shared)."""
    z = np.load(cache / f"{rec['name']}.npz")
    shape = tuple(z["shape"])
    rows, cols = shape[0] // F.CELL, shape[1] // F.CELL
    X = f16(z["X"].astype(np.float32)).astype(np.float32)
    out = {"shape": shape, "counts": z["counts"].astype(int), "cells": {}, "score": {}, "wup": {}}
    roi = F.cell_roi(z["fabric"], shape)
    for v, c in colsets.items():
        p = models[v].predict_proba(X[:, c])[:, 1] * roi
        out["cells"][v] = p
        out["score"][v] = float(np.sort(p)[-3:].mean())
        out["wup"][v] = upsample(p.reshape(rows, cols), shape)
    fabric = z["fabric"]
    cand = (np.max(list(out["wup"].values()), 0) >= TW[0]) & fabric[None, :]
    img = afid.read_image(rec)
    pprob = np.zeros(shape, np.float32)
    if cand.any():
        ch = F.channels(F.preprocess(img, fabric), float(z["px"]), float(z["py"]), fabric)
        pprob[cand] = models["pixel"].predict_proba(ch[:, cand].T)[:, 1]
    out["pprob"] = pprob
    out["gt"] = afid.read_mask(rec, shape)
    out["img"] = img
    return out


# --- Threshold search on inner out-of-fold images --------------------------------------------


def grid_counts(wup, pprob, gt):
    """Pixel TP and FP for every (TW, TP) pair; FN = GT total - TP."""
    tp = np.zeros((len(TW), len(TP)), np.int64)
    fp = np.zeros_like(tp)
    g = gt.astype(bool)
    edges = np.concatenate([TP, [1.01]])
    for i, t in enumerate(TW):
        r = wup >= t
        hp = np.histogram(pprob[r & g], edges)[0]
        hn = np.histogram(pprob[r & ~g], edges)[0]
        tp[i] = np.cumsum(hp[::-1])[::-1]
        fp[i] = np.cumsum(hn[::-1])[::-1]
    return tp, fp, int(g.sum())


def window_counts(p, counts):
    pos, neg = counts >= POS_MIN, counts == 0
    return np.array([[((p >= t) & pos).sum(), ((p >= t) & neg).sum(), ((p < t) & pos).sum()] for t in TW])


def best_pixel(tp, fp, ngt):
    f1 = 2 * tp / np.maximum(2 * tp + fp + (ngt - tp), 1)
    i, j = np.unravel_index(np.argmax(f1), f1.shape)
    return float(TW[i]), float(TP[j]), float(f1[i, j])


def best_image(scores, labels):
    """Image threshold maximising accuracy on out-of-fold scores (midpoint between scores)."""
    s = np.sort(np.unique(scores))
    cands = np.concatenate([[s[0] - 1e-6], (s[1:] + s[:-1]) / 2, [s[-1] + 1e-6]])
    acc = [((scores >= t) == labels).mean() for t in cands]
    return float(cands[int(np.argmax(acc))])


def choose(oof, recs_by):
    """Every threshold, per variant, from inner OOF records only."""
    th = {}
    for v in VARIANTS:
        names = list(oof)
        s = np.array([oof[n]["score"][v] for n in names])
        y = np.array([recs_by[n]["label"] for n in names])
        fab = np.array([recs_by[n]["fabric"] for n in names])
        tp = sum(oof[n]["grid"][v][0] for n in names)
        fp = sum(oof[n]["grid"][v][1] for n in names)
        ngt = sum(oof[n]["grid"][v][2] for n in names)
        tw, tpx, f1 = best_pixel(tp, fp, ngt)
        wc = sum(oof[n]["win"][v] for n in names)
        wf1 = 2 * wc[:, 0] / np.maximum(2 * wc[:, 0] + wc[:, 1] + wc[:, 2], 1)
        t = {"img": best_image(s, y), "tw": tw, "tp": tpx, "inner_pixel_f1": f1,
             "win": float(TW[int(np.argmax(wf1))]), "fabric": {},
             "cal": IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y),
             "ref0": np.sort(s[y == 0]), "ref1": np.sort(s[y == 1])}
        for f in sorted(set(fab)):
            m = fab == f
            ft = {"img": t["img"], "tw": tw, "tp": tpx}
            if (y[m] == 1).sum() >= 3 and (y[m] == 0).sum() >= 3:
                ft["img"] = best_image(s[m], y[m])
            if (y[m] == 1).sum() >= 3:
                sel = [n for n, k in zip(names, m) if k]
                ft["tw"], ft["tp"], _ = best_pixel(sum(oof[n]["grid"][v][0] for n in sel),
                                                   sum(oof[n]["grid"][v][1] for n in sel),
                                                   sum(oof[n]["grid"][v][2] for n in sel))
            t["fabric"][f] = ft
        th[v] = t
    return th


# --- Masks and metrics ----------------------------------------------------------------------


def drop_small(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_AREA
    return keep[lab]


def hough_link(mask, cand):
    """Join broken thin defects along near-horizontal/vertical Hough segments, within candidates."""
    lines = cv2.HoughLinesP(mask * 255, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=30)
    if lines is None:
        return mask
    L = np.zeros_like(mask)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if ang < 10 or ang > 80:
            cv2.line(L, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3)
    return mask | (L & cand)


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def mask_counts(pred, gt):
    """Pixel TP/FP/FN plus tolerance-based hits at +-2 and +-3 px."""
    p, g = pred.astype(bool), gt.astype(bool)
    out = {"tp": int((p & g).sum()), "fp": int((p & ~g).sum()), "fn": int((~p & g).sum())}
    for d in (2, 3):
        out[f"tolP{d}"] = int((p & cv2.dilate(gt, disk(d)).astype(bool)).sum())
        out[f"tolR{d}"] = int((g & cv2.dilate(pred, disk(d)).astype(bool)).sum())
    out["npred"], out["ngt"] = int(p.sum()), int(g.sum())
    return out


def conformal(s, t):
    p0 = (1 + np.sum(t["ref0"] >= s)) / (len(t["ref0"]) + 1)
    p1 = (1 + np.sum(t["ref1"] <= s)) / (len(t["ref1"]) + 1)
    pset = {k for k, p in (("normal", p0), ("defect", p1)) if p > ALPHA}
    label = "reliable" if len(pset) == 1 else ("uncertain" if pset else "atypical")
    return float(p0), float(p1), "+".join(sorted(pset)) or "empty", label


# --- One outer fold ------------------------------------------------------------------------


def run_fold(args):
    k, train, test, recs, cache, out = args
    done = out / f"fold{k}.pkl"
    if done.exists():
        return k
    by = {r["name"]: r for r in recs}
    names = json.loads((cache / "feature_names.json").read_text())
    colsets = {v: columns(names, v) for v in VARIANTS}
    pool = load_pool(recs, cache)
    strata = [f"{by[n]['label']}_{by[n]['fabric']}" for n in train]
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED + k)
    oof = {}
    for j, (a, b) in enumerate(inner.split(train, strata)):
        models = fit(pool, [train[i] for i in a], colsets)
        for i in b:
            p = predict(models, colsets, by[train[i]], cache)
            oof[train[i]] = {"score": p["score"],
                             "grid": {v: grid_counts(p["wup"][v], p["pprob"], p["gt"]) for v in VARIANTS},
                             "win": {v: window_counts(p["cells"][v], p["counts"]) for v in VARIANTS}}
        print(f"  fold {k}: inner {j + 1}/3 done", flush=True)
    th = choose(oof, by)
    models = fit(pool, train, colsets)
    maps = out / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in test:
        r = by[n]
        p = predict(models, colsets, r, cache)
        row = {"name": n, "fold": k, "label": r["label"], "fabric": r["fabric"], "defect": r["defect"],
               "variants": {}}
        for v in VARIANTS:
            t = th[v]
            s = p["score"][v]
            vr = {"score": s, "prob": float(t["cal"].predict([s])[0]), "win": {}}
            for mode, tt in (("global", t), ("fabric", t["fabric"].get(r["fabric"], t))):
                pred = int(s >= tt["img"])
                m = ((p["wup"][v] >= tt["tw"]) & (p["pprob"] >= tt["tp"])).astype(np.uint8) * pred
                vr[mode] = {"pred": pred, "raw": mask_counts(m, p["gt"])}
                if v == FINAL:
                    post = drop_small(m)
                    cand = ((p["wup"][v] >= tt["tw"] / 2) & (p["pprob"] >= tt["tp"] / 2)).astype(np.uint8) * pred
                    hough = hough_link(post, cand)
                    vr[mode]["post"] = mask_counts(post, p["gt"])
                    vr[mode]["hough"] = mask_counts(hough, p["gt"])
                    if mode == "global":
                        cv2.imwrite(str(maps / f"{n}_mask.png"), post * 255)
                        cv2.imwrite(str(maps / f"{n}_mask_hough.png"), hough * 255)
            vr["win"] = window_counts(p["cells"][v], p["counts"])[list(TW).index(t["win"])]
            vr["p_normal"], vr["p_defect"], vr["set"], vr["reliability"] = conformal(s, t)
            row["variants"][v] = vr
        dev = p["wup"][FINAL] * p["pprob"]
        cv2.imwrite(str(maps / f"{n}_deviation.png"), np.clip(dev * 255, 0, 255).astype(np.uint8))
        rows.append(row)
    thr = {v: {k2: t[k2] for k2 in ("img", "tw", "tp", "win", "inner_pixel_f1", "fabric")} for v, t in th.items()}
    with open(done, "wb") as f:
        pickle.dump({"rows": rows, "thresholds": thr,
                     "oof_scores": {n: o["score"] for n, o in oof.items()}}, f)
    print(f"  fold {k}: test done ({len(test)} images)", flush=True)
    return k


def cross_validate(recs, cache, out):
    out.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in recs]
    strata = [f"{r['label']}_{r['fabric']}" for r in recs]
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED).split(names, strata)
    jobs = [(k, [names[i] for i in a], [names[i] for i in b], recs, cache, out) for k, (a, b) in enumerate(folds)]
    # One process per fold, results on disk: no result pipe to break on a busy machine.
    running = []
    for job in jobs:
        while sum(p.is_alive() for p in running) >= WORKERS:
            time.sleep(5)
        running.append(Process(target=run_fold, args=(job,)))
        running[-1].start()
    for p in running:
        p.join()
    failed = [k for k in range(len(jobs)) if not (out / f"fold{k}.pkl").exists()]
    if failed:
        raise RuntimeError(f"folds {failed} did not finish; rerun to resume")


if __name__ == "__main__":
    # Threshold search and metrics on a toy case with a known answer.
    gt = np.zeros((32, 64), np.uint8)
    gt[10:12, 5:60] = 1
    pprob = gt * 0.9 + 0.02
    wup = np.full(gt.shape, 0.5, np.float32)
    tp, fp, ngt = grid_counts(wup, pprob.astype(np.float32), gt)
    tw, tpx, f1 = best_pixel(tp, fp, ngt)
    assert f1 == 1.0 and tw <= 0.5, (tw, tpx, f1)
    c = mask_counts(np.roll(gt, 1, 0), gt)
    assert c["tp"] == 55 and c["tolP2"] == c["npred"] and c["tolR2"] == c["ngt"]
    assert best_image(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 0.5
    print("pipeline ok")
