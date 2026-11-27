"""Bring your own data.

Three ways in, from least to most custom:

1. ``with_truth(problem, x)``: your image or signal as the ground truth of a built-in problem.
   The forward model and noise simulation are unchanged, so every metric still works.
2. ``with_measured(problem, b, sigma=None)``: real measurements for a built-in forward model.
   The truth is unknown (stored as NaN), so error / PSNR / SSIM are NaN and the oracle rule is off.
   The noise level is taken from ``sigma`` or estimated from the data (median absolute deviation).
3. ``custom_problem(A, b, ...)``: your own matrix (dense or scipy.sparse) and data vector.

Built-in operators are normalized to ||A|| = 1. Measured data are given in the physical units of the
un-normalized operator and are rescaled here, so x comes out in the units you expect.
Files: .npy, .npz, .csv, .txt, .mat, and images (.png, .jpg, .tif, .bmp) via ``load_array``.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import scipy.sparse as sp

from . import operators as ops
from .problems import Data, Problem

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")


# ----------------------------------------------------------------------------- loading
def load_array(path, key=None):
    """Read an array from disk. Images are converted to gray levels in [0, 1]."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return np.asarray(z[key] if key else z[z.files[0]])
    if ext in (".csv", ".txt", ".dat"):
        with open(path, encoding="utf-8") as fh:
            head = fh.read(4096)
        delim = "," if "," in head else (";" if ";" in head else None)
        return np.loadtxt(path, delimiter=delim, ndmin=1)
    if ext == ".mtx":
        import scipy.io
        return scipy.io.mmread(path)
    if ext == ".mat":
        import scipy.io
        m = scipy.io.loadmat(path)
        names = [k for k in m if not k.startswith("__")]
        if not names:
            raise ValueError(f"{path}: no arrays found")
        v = m[key] if key else m[names[0]]
        return v if sp.issparse(v) else np.asarray(v)
    if ext in IMAGE_EXT:
        import matplotlib.image as mpimg
        img = np.asarray(mpimg.imread(path), float)
        if img.ndim == 3:
            rgb = img[..., :3]
            img = rgb @ np.array([0.299, 0.587, 0.114])
        if img.max() > 1.0:
            img = img / 255.0
        return img
    raise ValueError(f"unsupported file type {ext!r}")


def resample(x, shape):
    """Resize an image (bilinear) or a signal (linear interpolation) to ``shape``."""
    x = np.asarray(x, float)
    shape = tuple(shape)
    if len(shape) == 2:
        if x.ndim == 1 and x.size == shape[0] * shape[1]:
            return x.reshape(shape)
        if x.ndim != 2:
            raise ValueError(f"an image problem needs a 2-D array, got shape {x.shape}")
        if x.shape == shape:
            return x
        from scipy.ndimage import zoom
        y = zoom(x, (shape[0] / x.shape[0], shape[1] / x.shape[1]), order=1)
        out = np.zeros(shape)
        h, w = min(shape[0], y.shape[0]), min(shape[1], y.shape[1])
        out[:h, :w] = y[:h, :w]
        if h < shape[0]:
            out[h:] = out[h - 1]
        if w < shape[1]:
            out[:, w:] = out[:, [w - 1]]
        return out
    x = x.ravel()
    n = shape[0]
    if x.size == n:
        return x
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, x.size), x)


# ----------------------------------------------------------------------------- noise estimate
def estimate_sigma(b, shape=None):
    """Robust per-entry noise std from the data alone (Donoho's MAD rule).

    Images: finest diagonal Haar detail. Signals: first differences. Assumes the clean data are smooth
    relative to the noise, which holds for blurred or band-limited measurements."""
    b = np.asarray(b, float)
    if shape is not None and len(shape) == 2 and b.size == shape[0] * shape[1] and min(shape) >= 4:
        z = b.reshape(shape)
        z = z[: z.shape[0] // 2 * 2, : z.shape[1] // 2 * 2]
        hh = (z[0::2, 0::2] - z[0::2, 1::2] - z[1::2, 0::2] + z[1::2, 1::2]) / 2.0
        return float(np.median(np.abs(hh)) / 0.6745)
    d = np.diff(b.ravel())
    return float(np.median(np.abs(d - np.median(d))) / (0.6745 * np.sqrt(2)))


def estimate_sigma_gcv(A, b):
    """Noise std from a GCV-tuned Tikhonov fit (Wahba's estimator): sigma^2 = ||r||^2 / (m - trace H).

    Works for any matrix up to a few thousand columns, smooth data or not."""
    A = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    b = np.asarray(b, float).ravel()
    m = b.size
    U_, s, _ = np.linalg.svd(A, full_matrices=False)
    beta = U_.T @ b
    oor = max(float(b @ b - beta @ beta), 0.0)
    best = None
    for lam in np.logspace(-12, 1, 261) * s[0] ** 2:
        phi = s ** 2 / (s ** 2 + lam)
        r2 = float(np.sum(((1 - phi) * beta) ** 2)) + oor
        dof = m - float(phi.sum())
        if dof <= 0.5:
            continue
        g = r2 / dof ** 2
        if best is None or g < best[0]:
            best = (g, r2 / dof)
    return float(np.sqrt(best[1])) if best else estimate_sigma(b)


def _level(b, sigma):
    """Relative noise level sigma sqrt(m) / ||A x||, with ||A x||^2 ~= ||b||^2 - m sigma^2."""
    m = b.size
    clean = np.sqrt(max(float(b @ b) - m * sigma ** 2, 0.01 * float(b @ b), 1e-300))
    return float(sigma * np.sqrt(m) / clean)


# ----------------------------------------------------------------------------- 1. your truth
def with_truth(problem: Problem, x, normalize=None, name="user") -> Problem:
    """Copy of ``problem`` with ``x`` (resampled to the problem grid) as ground truth.

    Images are rescaled to [0, 1] by default, matching the built-in phantoms."""
    xt = resample(x, problem.shape)
    if normalize is None:
        normalize = problem.is_image
    if normalize:
        lo, hi = float(np.nanmin(xt)), float(np.nanmax(xt))
        xt = (xt - lo) / (hi - lo) if hi > lo else np.zeros_like(xt)
    meta = dict(problem.meta, user_truth=name)
    nonneg = problem.nonneg and bool(np.nanmin(xt) >= 0)
    return dataclasses.replace(problem, x_true=np.asarray(xt, float).ravel(), meta=meta, nonneg=nonneg)


# ----------------------------------------------------------------------------- 2. your measurements
def _as_data_vector(problem: Problem, b):
    """Map the user's array to the problem's data vector (handles masks and k-space)."""
    b = np.asarray(b)
    m = problem.m
    if np.iscomplexobj(b) and problem.data_kind == "kspace":
        mask = problem.meta["mask"]
        if b.shape != mask.shape:
            raise ValueError(f"k-space data must have the mask shape {mask.shape} (unshifted, orthonormal FFT)")
        z = b.ravel()[np.flatnonzero(mask.ravel())]
        return np.concatenate([z.real, z.imag])
    b = np.asarray(b, float)
    if b.size == m:
        return b.ravel().copy()
    if problem.data_kind == "masked" and b.size == problem.n:
        return b.ravel()[problem.meta["idx"]].copy()
    raise ValueError(f"data has {b.size} entries; {problem.key} expects {m}"
                     + (f" (shape {problem.data_shape})" if problem.data_shape else ""))


def with_measured(problem: Problem, b, sigma=None, truth=None, name="user"):
    """Your measured data for a built-in forward model. Returns ``(problem, data)``.

    ``sigma`` is the per-entry noise std in the units of ``b``; when omitted it is estimated."""
    bv = _as_data_vector(problem, b)
    scale = float(problem.meta.get("A_scale", 1.0))
    bv = bv / scale
    if sigma is None:
        shape = problem.data_shape if problem.data_kind in ("image", "lowres") else None
        s = estimate_sigma(bv, shape)
        estimated = True
    else:
        s = float(sigma) / scale
        estimated = False
    m = bv.size
    level = _level(bv, s)
    meta = dict(problem.meta, user_data=name, sigma_estimated=estimated,
                lam_ref=problem.meta.get("lam_ref", problem.noise_default))
    xt = np.full(problem.n, np.nan) if truth is None else resample(truth, problem.shape).ravel()
    d = Data(bv, "measured", float(level), float(s), float(s * np.sqrt(m)))
    meta["measured"] = d
    p = dataclasses.replace(problem, x_true=np.asarray(xt, float), meta=meta, noise_kinds=("measured",),
                            noise_default=("measured", level))
    return p, d


# ----------------------------------------------------------------------------- 3. your matrix
def custom_problem(A, b, x_true=None, shape=None, sigma=None, nonneg=None, title="Custom problem (A, b)"):
    """A linear problem from your own matrix and data. Returns ``(problem, data)``.

    ``shape`` reshapes x for display (e.g. (64, 64) for an image). A is normalized to ||A|| = 1 and b
    rescaled with it, so x is unchanged."""
    if not sp.issparse(A):
        A = np.atleast_2d(np.asarray(A, float))
    b = np.asarray(b, float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"A is {m} x {n} but b has {b.size} entries")
    shape = tuple(shape) if shape else (n,)
    if int(np.prod(shape)) != n:
        raise ValueError(f"shape {shape} does not match n = {n}")
    op = ops.from_matrix(A, name="user matrix")
    s = op.norm()
    op = op.scaled(1.0 / s)
    op._norm = 1.0
    if nonneg is None:
        nonneg = x_true is not None and bool(np.min(x_true) >= 0)
    xt = np.full(n, np.nan) if x_true is None else np.asarray(x_true, float).ravel()
    if xt.size != n:
        raise ValueError(f"x_true has {xt.size} entries, A has {n} columns")
    bv = b / s
    if sigma is not None:
        sig = float(sigma) / s
    elif n <= 4000:
        sig = estimate_sigma_gcv(A, b) / s
    else:
        sig = estimate_sigma(bv)
    level = _level(bv, sig)
    p = Problem("custom", title, "User data", shape, xt, A=op, data_kind="signal", data_shape=(m,),
                equations=[r"b = A\,x + \varepsilon", r"A \in \mathbb{R}^{%d \times %d}" % (m, n)],
                note="Your matrix and data. A is scaled to unit norm (and b with it).",
                noise_kinds=("measured",), noise_default=("measured", level), nonneg=bool(nonneg),
                meta={"A_scale": s, "user_data": True, "sigma_estimated": sigma is None},
                x_label="index")
    d = Data(bv, "measured", float(level), float(sig), float(sig * np.sqrt(m)))
    p.meta["measured"] = d
    return p, d

