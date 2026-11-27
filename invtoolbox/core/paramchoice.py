"""Regularization-parameter choice: lambda sweeps and the classical selection rules.

Rules (all a-posteriori except the oracle, which uses the ground truth and is shown as a reference):
  discrepancy  Morozov: largest lambda with ||A x_lam - b|| <= tau * delta
  l-curve      corner of (log ||A x - b||, log R(x)): maximum signed curvature (Hansen)
  gcv          generalized cross validation, closed form for spectral Tikhonov
  quasi-opt    minimizer of ||x_{lam_{k+1}} - x_{lam_k}|| on a geometric grid
  oracle       minimum relative error against the truth
"""
from __future__ import annotations

import numpy as np

from . import methods as M
from .operators import Grad
from .prox import tv
from .spectral import Spectral
from .wavelets import Wavelet


def reg_value(method_key, params, problem, x):
    sh = problem.shape
    if method_key in ("tikhonov", "gaussian-posterior", "gauss-newton", "lbfgs"):
        L, _ = M._penalty_ops(problem, params.get("penalty", "identity"))
        return float(np.linalg.norm(L(x - problem.x0())))
    if method_key in ("ista", "fista"):
        basis = params.get("basis", "pixel")
        if basis == "pixel":
            return float(np.abs(x).sum())
        W = Wavelet(sh, basis)
        return float(np.abs(W.forward(x)[W.detail_mask()]).sum())
    if method_key == "admm-l1":
        return float(np.abs(x).sum())
    if method_key == "irls":
        p = params.get("p", 0.5)
        L, _ = M._penalty_ops(problem, "identity" if params.get("operator", "identity") == "identity" else "gradient")
        return float(np.sum(np.abs(L(x)) ** p))
    if method_key == "pdhg":
        r = params.get("reg", "TV")
        if r == "L1":
            return float(np.abs(x).sum())
        if r == "wavelet":
            W = Wavelet(sh, "db2")
            return float(np.abs(W.forward(x)[W.detail_mask()]).sum())
        return tv(x, sh)
    return tv(x, sh)


def lam_grid(center, decades=3.0, n=13):
    c = np.log10(max(center, 1e-12))
    return 10.0 ** np.linspace(c - decades, c + decades, n)


def sweep(method_key, problem, data, params=None, lams=None, stop=None, progress=None):
    params = dict(params or {})
    meth = M.METHODS[method_key]
    if lams is None:
        from .defaults import default_lam
        kw = meth.defaults()
        kw.update(params)
        lams = lam_grid(default_lam(method_key, problem, data, kw))
    lams = np.asarray(lams, float)
    out = {"lam": lams, "res": [], "reg": [], "err": [], "x": []}
    for i, lam in enumerate(lams):
        if stop is not None and stop():
            raise M.Stopped
        r = M.run(method_key, problem, data, dict(params, lam=float(lam)), stop=stop)
        out["res"].append(r.info["residual"])
        out["reg"].append(reg_value(method_key, r.params, problem, r.x))
        out["err"].append(r.info["rel_err"])
        out["x"].append(r.x)
        if progress is not None:
            progress(i, len(lams), out)
    for k in ("res", "reg", "err"):
        out[k] = np.asarray(out[k])
    if method_key == "tikhonov" and params.get("penalty", "identity") == "identity" and Spectral.available(problem):
        S = Spectral(problem)
        fine = np.logspace(np.log10(lams[0]), np.log10(lams[-1]), 200)
        out["spectral"] = S.curves(data.b, fine)
        out["gcv"] = S.curves(data.b, lams)["gcv"]
    out["choices"] = choose_all(out, data)
    return out


# ----------------------------------------------------------------------------- rules
def discrepancy(lams, res, delta, tau=1.02):
    ok = np.flatnonzero(res <= tau * delta)
    if ok.size == 0:
        return int(np.argmin(res))
    i = int(ok.max())
    return i


def lcurve_corner(res, reg):
    rho = np.log(np.maximum(res, 1e-300))
    eta = np.log(np.maximum(reg, 1e-300))
    if len(rho) < 5:
        return int(len(rho) // 2)
    t = np.arange(len(rho), dtype=float)
    d1r, d1e = np.gradient(rho, t), np.gradient(eta, t)
    d2r, d2e = np.gradient(d1r, t), np.gradient(d1e, t)
    kappa = (d1r * d2e - d2r * d1e) / np.maximum((d1r ** 2 + d1e ** 2) ** 1.5, 1e-300)
    kappa[[0, -1]] = -np.inf
    return int(np.argmax(kappa))


def quasi_optimality(xs):
    if len(xs) < 3:
        return 0
    d = [np.linalg.norm(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    return int(np.argmin(d))


def choose_all(sw, data, tau=1.02):
    lams = sw["lam"]
    ch = {
        "discrepancy": discrepancy(lams, sw["res"], data.delta, tau),
        "l-curve": lcurve_corner(sw["res"], sw["reg"]),
        "quasi-opt": quasi_optimality(sw["x"]),
    }
    if np.isfinite(sw["err"]).any():       # only when a ground truth is known
        ch["oracle"] = int(np.nanargmin(sw["err"]))
    if "gcv" in sw:
        ch["gcv"] = int(np.argmin(sw["gcv"]))
    return ch


def iteration_choice(hist, delta, tau=1.02):
    """For iterative regularization: discrepancy stopping index and oracle index from one run."""
    res, err = np.asarray(hist["res"]), np.asarray(hist["err"])
    if res.size == 0:
        return {}
    ok = np.flatnonzero(res <= tau * delta)
    out = {"discrepancy": int(ok[0]) if ok.size else int(res.size - 1)}
    if np.isfinite(err).any():
        out["oracle"] = int(np.nanargmin(err))
    return out
