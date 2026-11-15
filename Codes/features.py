"""Classical texture descriptors for AFID strips (no deep learning).

Stages, all unsupervised and computed per strip (so they never see labels or other images):
  1. preprocess: illumination flattening (subtract a sigma-25 Gaussian), 3x3 median, CLAHE.
  2. weave_period: dominant thread period along x and y from the row/column power spectrum.
     It sets the *adaptive* Gabor wavelengths and GLCM distance (fixed alternative: Week-7 lambda=10).
  3. channels: per-pixel filter bank (median deviation, local std, row/column line profiles,
     LoG at two scales, Gabor energy), each robustly z-scored within the strip (median / MAD).
  4. window_features: descriptors on a 16-px cell grid, for 32 and 64 px windows centred on each
     cell: intensity, LBP riu2 histograms (R=1 P=8, R=2 P=16), GLCM statistics, Gabor energy
     (fixed and adaptive banks), filter-bank mean/max. Each also gets a strip-normalised copy.
"""

import zlib

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_laplace
from skimage.feature import local_binary_pattern

CELL = 16
SCALES = (32, 64)
FIXED_LAMBDAS = (10.0, 20.0)  # Week-7 notebook used lambda = 10
THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
CHANNELS = ("med", "std", "row", "col", "log2", "log4", "gab", "blob", "rowabs", "colabs", "ctx")
BASE_CHANNELS = 7  # the first seven are aggregated into window features


def fabric_roi(img):
    """Fabric columns (bool, W): drops the bright background band beside the fabric edge (> 50 % of the
    column >= 245), with a 4 px margin; 158 of 245 AFID strips have one, up to 30 % of the width."""
    bg = (img >= 245).mean(0) > 0.5
    bg = np.convolve(bg.astype(float), np.ones(9), "same") > 0
    return ~bg if (~bg).sum() >= 256 else np.ones(img.shape[1], bool)


def rz(x, cols):
    """Robust z-score over the fabric columns of the strip: (x - median) / (1.4826 MAD)."""
    s = x[::4][:, cols][:, ::4]
    m = float(np.median(s))
    mad = 1.4826 * float(np.median(np.abs(s - m)))
    if mad < 1e-6:
        mad = float(s.std()) + 1e-6
    return (x - m) / mad


def preprocess(img, cols):
    """Background band filled with the fabric median first, so flattening does not smear it."""
    f = img.astype(np.float32)
    f[:, ~cols] = np.median(f[:, cols])
    flat = np.clip(f - cv2.GaussianBlur(f, (0, 0), 25) + 128, 0, 255).astype(np.uint8)
    flat = cv2.medianBlur(flat, 3)
    tiles = (max(1, img.shape[1] // 128), 2)
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=tiles).apply(flat)


def _period(f, axis):
    spec = (np.abs(np.fft.rfft(f, axis=axis)) ** 2).mean(axis=1 - axis)
    n = f.shape[axis]
    k = np.arange(len(spec))
    ok = (k > 0) & (n / np.maximum(k, 1) >= 3) & (n / np.maximum(k, 1) <= 48)
    return float(n / k[ok][np.argmax(spec[ok])])


def weave_period(P, cols):
    """Dominant texture period in px along x and along y (3 to 48 px), from the fabric columns."""
    f = P[:, cols].astype(np.float32)
    f -= f.mean()
    return _period(f, 1), _period(f, 0)


def gabor_energy(P, lam, theta):
    sigma = 0.56 * lam
    k = int(2 * np.ceil(2.5 * sigma) + 1)
    re = cv2.getGaborKernel((k, k), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
    im = cv2.getGaborKernel((k, k), sigma, theta, lam, 0.5, np.pi / 2, ktype=cv2.CV_32F)
    re -= re.mean()
    f = P.astype(np.float32)
    return cv2.magnitude(cv2.filter2D(f, cv2.CV_32F, re), cv2.filter2D(f, cv2.CV_32F, im))


def adaptive_lambdas(px, py):
    """Wavelength per orientation: x-period at 0, y-period at 90, their mean on the diagonals."""
    base = (px, (px + py) / 2, py, (px + py) / 2)
    return [float(np.clip(b, 4, 32)) for b in base]


def channels(P, px, py, cols):
    """Per-pixel filter bank, (11, H, W) float32 robust z-scores, names in CHANNELS."""
    f = P.astype(np.float32)
    med = f - cv2.medianBlur(P, 21)
    m = cv2.blur(f, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (7, 7)) - m * m, 0))
    hb = cv2.blur(f, (61, 1))
    row = hb - gaussian_filter1d(hb, 5, axis=0)
    vb = cv2.blur(f, (1, 31))
    col = vb - gaussian_filter1d(vb, 5, axis=1)
    log2 = -gaussian_laplace(f, 2)
    log4 = -gaussian_laplace(f, 4)
    gab = np.max([rz(gabor_energy(P, lam, t), cols) for lam, t in zip(adaptive_lambdas(px, py), THETAS)], 0)
    z = [rz(c, cols) for c in (med, std, row, col, log2, log4)] + [gab]
    for c in z:
        c[:, ~cols] = 0
    a = [np.abs(c) for c in z[:4]] + [np.abs(z[4])]
    blob = cv2.GaussianBlur(a[0], (0, 0), 6)
    rowabs = cv2.GaussianBlur(a[2], (0, 0), 3)
    colabs = cv2.GaussianBlur(a[3], (0, 0), 3)
    ctx = cv2.GaussianBlur(np.max([a[0], a[2], a[3], a[4]], 0), (0, 0), 12)
    return np.stack(z + [blob, rowabs, colabs, ctx]).astype(np.float32)


def grid(shape):
    """Cell-centre rows and columns of the 16-px grid (partial last cell dropped)."""
    h, w = shape
    return np.arange(h // CELL) * CELL + CELL // 2, np.arange(w // CELL) * CELL + CELL // 2


def window_features(P, ch, px, py, fabric):
    """(n_cells, F) float32 matrix and feature names; names are 'scale:group:name'."""
    cy, cx = grid(P.shape)
    cols, names = [], []

    def add(x, s, grp, nm, pool="mean"):
        k = np.ones((s, s), np.uint8)
        y = cv2.dilate(x, k) if pool == "max" else cv2.blur(x, (s, s), borderType=cv2.BORDER_REFLECT)
        cols.append(y[cy][:, cx].ravel())
        names.append(f"s{s}:{grp}:{nm}")

    f = P.astype(np.float32)
    lbp1 = local_binary_pattern(P, 8, 1, "uniform").astype(np.uint8)
    lbp2 = local_binary_pattern(P, 16, 2, "uniform").astype(np.uint8)
    q = f / 16.0
    dA = (max(2, round(px)), max(2, round(py)))
    for s in SCALES:
        add(f, s, "int", "mean")
        m = cv2.blur(f, (s, s))
        add(np.sqrt(np.maximum(cv2.blur(f * f, (s, s)) - m * m, 0)), 1, "int", "std")
        names[-1] = f"s{s}:int:std"
        for b in range(10):
            add((lbp1 == b).astype(np.float32), s, "lbp", f"r1_{b}")
        for b in range(18):
            add((lbp2 == b).astype(np.float32), s, "lbp", f"r2_{b}")
        mu = cv2.blur(q, (s, s))
        var = np.maximum(cv2.blur(q * q, (s, s)) - mu * mu, 1e-3)
        for grp, dists in (("glcm1", (1, 1)), ("glcmA", dA)):
            for ang, d in zip((0, 90), dists):
                b = np.roll(q, -d, axis=1 if ang == 0 else 0)
                diff = q - b
                add(diff * diff, s, grp, f"contrast{ang}")
                add(np.abs(diff), s, grp, f"dissim{ang}")
                add(1 / (1 + diff * diff), s, grp, f"homog{ang}")
                cols.append(((cv2.blur(q * b, (s, s)) - mu * mu) / var)[cy][:, cx].ravel())
                names.append(f"s{s}:{grp}:corr{ang}")
        for grp, lams in (("gabF", None), ("gabA", adaptive_lambdas(px, py))):
            for mult in (1, 2):
                for i, t in enumerate(THETAS):
                    lam = FIXED_LAMBDAS[mult - 1] if lams is None else min(lams[i] * mult, 48)
                    add(gabor_energy(P, lam, t), s, grp, f"l{mult}_t{i}")
        for i in range(BASE_CHANNELS):
            a = np.abs(ch[i])
            add(a, s, "chan", f"{CHANNELS[i]}_mean")
            add(a, s, "chan", f"{CHANNELS[i]}_max", pool="max")
    X = np.stack(cols, 1).astype(np.float32)
    ref = X[cell_roi(fabric, P.shape)]
    med = np.median(ref, 0)
    mad = 1.4826 * np.median(np.abs(ref - med), 0)
    mad = np.where(mad < 1e-6, ref.std(0) + 1e-6, mad)
    return np.hstack([X, (X - med) / mad]), names + ["z|" + n for n in names]


def cell_roi(fabric, shape):
    """Cells whose centre column is fabric, flattened like the feature rows."""
    cy, cx = grid(shape)
    return np.repeat(fabric[cx][None], len(cy), 0).ravel()


def cell_counts(gt):
    """Ground-truth pixel count per 16x16 cell."""
    h, w = (gt.shape[0] // CELL) * CELL, (gt.shape[1] // CELL) * CELL
    return gt[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL).sum((1, 3)).ravel()


def sample_pixels(ch, gt, name, fabric):
    """Fixed-seed pixel sample for the pixel model: GT pixels, hard negatives near GT, random negatives;
    negatives only on fabric."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    h, w = gt.shape
    pos = np.flatnonzero(gt)
    near = cv2.dilate(gt, np.ones((51, 51), np.uint8)) & ~cv2.dilate(gt, np.ones((5, 5), np.uint8)) & 1
    near[:, ~fabric] = 0
    picks = [rng.choice(pos, min(1500, len(pos)), replace=False)] if len(pos) else []
    hard = np.flatnonzero(near)
    if len(hard):
        picks.append(rng.choice(hard, min(1000, len(hard)), replace=False))
    rand = rng.integers(0, h * w, 2000)
    picks.append(rand[fabric[rand % w]][:1000 if len(pos) else 800])
    idx = np.concatenate(picks)
    return ch.reshape(len(CHANNELS), -1)[:, idx].T.astype(np.float16), gt.ravel()[idx].astype(np.uint8)


def analyse(img, windows=True):
    """Everything per strip: fabric ROI, preprocessed image, periods, channels, window features."""
    fabric = fabric_roi(img)
    P = preprocess(img, fabric)
    px, py = weave_period(P, fabric)
    ch = channels(P, px, py, fabric)
    out = {"fabric": fabric, "P": P, "px": px, "py": py, "ch": ch}
    if windows:
        out["X"], out["names"] = window_features(P, ch, px, py, fabric)
    return out


if __name__ == "__main__":
    # Synthetic weave with period 8 x 6 and one dark horizontal line: periods recovered, line flagged.
    yy, xx = np.mgrid[:256, :1024]
    img = (128 + 40 * np.sin(2 * np.pi * xx / 8) * np.sin(2 * np.pi * yy / 6)).astype(np.uint8)
    img[120:123, 300:700] = 40
    img[:, 900:] = 255  # background band beside the fabric
    a = analyse(img)
    assert a["fabric"][:880].all() and not a["fabric"][900:].any()
    assert abs(a["px"] - 8) < 0.6 and abs(a["py"] - 6) < 0.6, (a["px"], a["py"])
    row = a["ch"][CHANNELS.index("row")]
    assert np.abs(row[121, 400:600]).mean() > 3 * np.abs(row[30, 400:600]).mean()
    print("features ok:", a["X"].shape, f"period {a['px']:.1f} x {a['py']:.1f}")
