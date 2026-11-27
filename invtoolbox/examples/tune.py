"""Find error-optimal default lambdas for every (problem, method, variant) at default settings.

    python -m invtoolbox.examples.tune [--jobs 4]

Writes invtoolbox/core/lam_defaults.json. Coarse decade grid, then a quarter-decade refinement.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

VARIANTS = {
    "tikhonov": [{"penalty": v} for v in ("identity", "gradient", "laplacian")],
    "ista": [{"basis": v} for v in ("pixel", "haar", "db2", "db4")],
    "fista": [{"basis": v} for v in ("pixel", "haar", "db2", "db4")],
    "pdhg": [{"data_term": d, "reg": r} for d in ("L2", "L1", "KL") for r in ("TV", "TGV2", "L1", "wavelet")],
    "irls": [{"operator": v} for v in ("identity", "gradient")],
    "admm-l1": [{}], "admm-tv": [{}], "richardson-lucy": [{}],
    "gauss-newton": [{"penalty": v} for v in ("identity", "gradient", "laplacian")],
    "lbfgs": [{"penalty": v} for v in ("identity", "gradient", "laplacian")],
}
COPY = {"gaussian-posterior": ("tikhonov", "penalty"), "myula-tv": ("pdhg", "L2,TV")}


def _err(args):
    from invtoolbox.core import methods as M, problems as P
    key, mk, var, lam = args
    p = P.build(key)
    d = p.simulate(seed=0)
    try:
        r = M.run(mk, p, d, dict(var, lam=lam))
        e = r.info["rel_err"]
        return e if np.isfinite(e) else 1e9
    except Exception:
        return 1e9


def tune_one(job):
    key, mk, var = job
    lo, hi = (-7, 1) if mk != "richardson-lucy" else (-6, -1)
    grid = [10.0 ** e for e in range(lo, hi + 1)]
    errs = [_err((key, mk, var, l)) for l in grid]
    i = int(np.argmin(errs))
    c = np.log10(grid[i])
    fine = [10 ** (c + d) for d in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)]
    ferrs = [_err((key, mk, var, l)) for l in fine]
    allg, alle = grid + fine, errs + ferrs
    j = int(np.argmin(alle))
    return job, float(allg[j]), float(alle[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", default="")
    ap.add_argument("--methods", default="")
    a = ap.parse_args()
    from invtoolbox.core import methods as M, problems as P
    from invtoolbox.core.defaults import _PATH, variant
    jobs = []
    for key in P.REGISTRY:
        if a.only and key not in a.only.split(","):
            continue
        p = P.build(key)
        ok = M.applicable(p)
        for mk, vars_ in VARIANTS.items():
            if ok.get(mk) or (a.methods and mk not in a.methods.split(",")):
                continue
            for v in vars_:
                if v.get("data_term") == "KL" and not p.A.nonneg:
                    continue
                jobs.append((key, mk, v))
    out = {}
    if os.path.exists(_PATH):
        with open(_PATH) as fh:
            out = json.load(fh)
    with Pool(a.jobs) as pool:
        for (key, mk, v), lam, err in pool.imap_unordered(tune_one, jobs):
            vs = variant(mk, v)
            out[f"{key}|{mk}|{vs}"] = lam
            print(f"{key:22s} {mk:16s} {vs:14s} lam={lam:.3g} err={err:.3f}", flush=True)
    for key in {k.split("|")[0] for k in out}:
        for dst, (src, how) in COPY.items():
            for k, v in list(out.items()):
                kk, mk, vs = k.split("|")
                if kk == key and mk == src and (how == "penalty" or vs == how):
                    out[f"{key}|{dst}|{vs if how == 'penalty' else '-'}"] = v
    with open(_PATH, "w") as fh:
        json.dump(dict(sorted(out.items())), fh, indent=0)


if __name__ == "__main__":
    main()
