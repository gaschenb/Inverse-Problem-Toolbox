"""Default regularization weights.

LAM[(problem, method, variant)] holds the error-optimal lambda at the problem's default noise level and
size, found by ``python -m invtoolbox.examples.tune``. At other noise levels the value is rescaled by
(level / default level) ** method.lam_power (2 for quadratic priors, 1 otherwise).
"""
from __future__ import annotations

import json
import os

_PATH = os.path.join(os.path.dirname(__file__), "lam_defaults.json")
try:
    with open(_PATH) as fh:
        LAM = {tuple(k.split("|")): v for k, v in json.load(fh).items()}
except FileNotFoundError:
    LAM = {}

FALLBACK = {"tikhonov": 1e-2, "gaussian-posterior": 1e-2, "gauss-newton": 1e-3, "lbfgs": 1e-3}


def variant(method_key, kw):
    keys = {"tikhonov": ("penalty",), "gaussian-posterior": ("penalty",), "gauss-newton": ("penalty",),
            "lbfgs": ("penalty",), "ista": ("basis",), "fista": ("basis",), "pdhg": ("data_term", "reg"),
            "irls": ("operator",)}.get(method_key, ())
    return ",".join(str(kw.get(k)) for k in keys) or "-"


def default_lam(method_key, problem, data, kw):
    from .methods import METHODS
    v = variant(method_key, kw)
    lam = LAM.get((problem.key, method_key, v))
    if lam is None:
        lam = LAM.get((problem.key, method_key, "-"), FALLBACK.get(method_key, 1e-3))
    kind0, lev0 = problem.meta.get("lam_ref", problem.noise_default)
    kind = "gaussian" if data is not None and data.kind == "measured" else getattr(data, "kind", None)
    if data is not None and kind == kind0 and lev0 > 0 and data.level > 0:
        lam *= (data.level / lev0) ** METHODS[method_key].lam_power
    return float(lam)
