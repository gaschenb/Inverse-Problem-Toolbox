"""Proximal maps, projections and denoisers shared by the solvers."""
from __future__ import annotations

import numpy as np

from .operators import Grad


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def project_ball(p, radius, iso=True):
    """Pointwise projection of a field p (channels first) onto {|p_i| <= radius}."""
    if iso:
        nrm = np.sqrt((p * p).sum(axis=0, keepdims=True))
        return p / np.maximum(1.0, nrm / max(radius, 1e-300))
    return np.clip(p, -radius, radius)


def tv(x, shape, iso=True):
    g = Grad(shape)(x)
    return float(np.sqrt((g * g).sum(0)).sum()) if iso else float(np.abs(g).sum())


def tv_prox(v, w, shape, n_iter=30, p0=None, iso=True):
    """argmin_x 0.5||x - v||^2 + w TV(x) by FGP (Beck-Teboulle) on the dual. Returns (x, p)."""
    D = Grad(shape)
    if w <= 0:
        return v.copy(), p0
    L = D.norm_sq()
    p = np.zeros((D.ndim,) + D.shape) if p0 is None else p0.copy()
    q = p.copy()
    t = 1.0
    for _ in range(n_iter):
        x = v - w * D.T(q)
        p_new = project_ball(q + D(x) / (w * L), 1.0, iso)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        q = p_new + (t - 1) / t_new * (p_new - p)
        p, t = p_new, t_new
    return v - w * D.T(p), p


def cg(apply_M, rhs, x0=None, n_iter=50, tol=1e-8):
    """Conjugate gradients for SPD M; relative residual tolerance."""
    x = np.zeros_like(rhs) if x0 is None else x0.copy()
    r = rhs - apply_M(x)
    p = r.copy()
    rs = float(r @ r)
    stop = (tol * max(np.linalg.norm(rhs), 1e-300)) ** 2
    for _ in range(n_iter):
        if rs <= stop:
            break
        Mp = apply_M(p)
        a = rs / max(float(p @ Mp), 1e-300)
        x += a * p
        r -= a * Mp
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def denoise(v, shape, kind, strength, state=None):
    """Plug-and-play denoisers operating on flattened arrays."""
    from scipy import ndimage
    from .wavelets import ti_denoise
    if kind == "tv":
        x, p = tv_prox(v, strength, shape, n_iter=25, p0=state)
        return x, p
    if kind == "wavelet":
        return ti_denoise(v, shape, strength, "db2", shifts=4 if len(shape) == 2 else 8), None
    if kind == "gaussian":
        return ndimage.gaussian_filter(v.reshape(shape), max(10 * strength, 1e-3)).ravel(), None
    if kind == "median":
        size = 3 if strength < 0.1 else 5
        return ndimage.median_filter(v.reshape(shape), size=size).ravel(), None
    raise ValueError(kind)
