"""Reconstruction quality metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def rel_err(x, problem):
    x = problem.align(np.asarray(x).ravel())
    return float(np.linalg.norm(x - problem.x_true) / max(np.linalg.norm(problem.x_true), 1e-300))


def psnr(x, problem):
    x = problem.align(np.asarray(x).ravel())
    xt = problem.x_true
    rng = max(float(xt.max() - xt.min()), 1e-12)
    mse = float(np.mean((x - xt) ** 2))
    return float(10 * np.log10(rng ** 2 / max(mse, 1e-30)))


def ssim(x, problem):
    """Gaussian-window SSIM (Wang et al. 2004), images only; NaN for signals."""
    if not problem.is_image:
        return float("nan")
    a = problem.align(np.asarray(x).ravel()).reshape(problem.shape)
    b = problem.x_true.reshape(problem.shape)
    L = max(float(b.max() - b.min()), 1e-12)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    f = lambda z: gaussian_filter(z, 1.5, truncate=3.5)
    ma, mb = f(a), f(b)
    saa = f(a * a) - ma ** 2
    sbb = f(b * b) - mb ** 2
    sab = f(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (saa + sbb + C2))
    return float(s.mean())


def summary(x, problem, data):
    r = problem.forward(x) - data.b
    return {
        "rel_err": rel_err(x, problem),
        "psnr": psnr(x, problem),
        "ssim": ssim(x, problem),
        "residual": float(np.linalg.norm(r)),
        "discrepancy": float(np.linalg.norm(r) / max(data.delta, 1e-300)),
    }
