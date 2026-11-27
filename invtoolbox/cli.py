"""Command-line interface.

    invtoolbox list
    invtoolbox solve   -p deblur-motion -m admm-tv [--lam 1e-3 | --lam discrepancy|l-curve|gcv|oracle]
                       [--set iters=200 --set iso=false] [--noise gaussian:0.02] [--size 96] [--truth text]
                       [--seed 1] [--out out/]
    invtoolbox compare -p ct-sparse-view -m fbp,cgls,sirt,pdhg [--out out/]

Your own data (see core/userdata.py):
    invtoolbox solve -p deblur-gaussian -m admm-tv --truth-file photo.png        your image as the truth
    invtoolbox solve -p deblur-gaussian -m admm-tv --data-file blurred.png [--sigma 0.01]   measured data
    invtoolbox solve --matrix A.npy --data-file b.csv -m tikhonov --lam gcv [--shape 64,64] [--truth-file x.npy]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .core import methods as M
from .core import paramchoice as PC
from .core import problems as P
from .core import userdata as U


def _parse_value(spec, raw):
    if spec.kind == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    if spec.kind == "int":
        return int(float(raw))
    if spec.kind == "float":
        return float(raw)
    return raw


def _params(method_key, sets, strict=True):
    specs = {p.key: p for p in M.METHODS[method_key].params}
    out = {}
    for s in sets or []:
        k, _, v = s.partition("=")
        if k not in specs and not strict:
            continue
        if k not in specs:
            raise SystemExit(f"{method_key} has no parameter {k!r}; available: {', '.join(specs)}")
        out[k] = _parse_value(specs[k], v)
    return out


def _setup(a):
    if a.matrix:
        if not a.data_file:
            raise SystemExit("--matrix needs --data-file")
        shape = tuple(int(v) for v in a.shape.split(",")) if a.shape else None
        xt = U.load_array(a.truth_file).ravel() if a.truth_file else None
        return U.custom_problem(U.load_array(a.matrix), U.load_array(a.data_file), xt, shape, a.sigma)
    if not a.problem:
        raise SystemExit("give a problem with -p (see `invtoolbox list`) or your own matrix with --matrix")
    if a.problem not in P.REGISTRY:
        raise SystemExit(f"unknown problem {a.problem!r}; see `invtoolbox list`")
    p = P.build(a.problem, a.size, a.truth)
    if a.truth_file:
        p = U.with_truth(p, U.load_array(a.truth_file), name=os.path.basename(a.truth_file))
    if a.data_file:
        return U.with_measured(p, U.load_array(a.data_file), a.sigma,
                               truth=p.x_true.reshape(p.shape) if a.truth_file else None,
                               name=os.path.basename(a.data_file))
    kind, level = None, None
    if a.noise:
        kind, _, lv = a.noise.partition(":")
        level = float(lv) if lv else None
    d = p.simulate(kind, level, a.seed)
    return p, d


def _solve(p, d, mk, params, lam_rule=None, verbose=True):
    if lam_rule:
        sw = PC.sweep(mk, p, d, params)
        i = sw["choices"].get(lam_rule)
        if i is None:
            raise SystemExit(f"rule {lam_rule!r} unavailable here; have {', '.join(sw['choices'])}")
        params = dict(params, lam=float(sw["lam"][i]))
        if verbose:
            print(f"  {lam_rule}: lambda = {params['lam']:.3g}")
    return M.run(mk, p, d, params)


def _report(r):
    i = r.info
    if not np.isfinite(i["rel_err"]):
        s = f"{r.method:20s} (no ground truth)"
    else:
        s = f"{r.method:20s} rel.err {i['rel_err']:.4f}  PSNR {i['psnr']:6.2f}"
    if np.isfinite(i["ssim"]):
        s += f"  SSIM {i['ssim']:.3f}"
    s += f"  |r|/delta {i['discrepancy']:.3f}  {r.seconds:.2f}s"
    if "lam" in r.params:
        s += f"  lambda {r.params['lam']:.3g}"
    return s


def _save(p, d, results, out):
    import matplotlib
    matplotlib.use("Agg")
    from .core.plots import figure_results
    os.makedirs(out, exist_ok=True)
    fig = figure_results(p, d, results)
    fig.savefig(os.path.join(out, f"{p.key}.png"), dpi=150)
    js = {"problem": p.key, "n": p.n, "m": p.m, "noise": [d.kind, d.level, d.seed], "delta": d.delta,
          "runs": [{"method": r.method, "params": {k: v for k, v in r.params.items()},
                    "metrics": {k: v for k, v in r.info.items() if isinstance(v, (int, float, str))},
                    "seconds": r.seconds} for r in results]}
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(js, fh, indent=2, default=str)
    for r in results:
        np.save(os.path.join(out, f"{p.key}_{r.method}.npy"), r.x.reshape(p.shape))
    print(f"wrote {out}/")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="invtoolbox", description="Inverse Problem Toolbox")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list problems and methods")
    for name in ("solve", "compare"):
        s = sub.add_parser(name)
        s.add_argument("-p", "--problem")
        s.add_argument("-m", "--method", required=name == "solve", default="all",
                       help="method key (compare: comma list or 'all')")
        s.add_argument("--lam", default=None, help="number or rule: discrepancy, l-curve, gcv, quasi-opt, oracle")
        s.add_argument("--set", action="append", help="method parameter key=value (repeatable)")
        s.add_argument("--noise", help="kind[:level], e.g. poisson:0.1")
        s.add_argument("--size", type=int)
        s.add_argument("--truth")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out")
        s.add_argument("--truth-file", help="your image / signal as ground truth (.png .npy .csv .mat ...)")
        s.add_argument("--data-file", help="your measured data for the chosen problem (or for --matrix)")
        s.add_argument("--sigma", type=float, help="noise std per data entry (default: estimated from the data)")
        s.add_argument("--matrix", help="your own forward matrix A (.npy .npz .csv .mtx .mat); with --data-file")
        s.add_argument("--shape", help="display shape of x for --matrix, e.g. 64,64")
    a = ap.parse_args(argv)
    if a.cmd in (None, "list"):
        for cat, keys in P.categories().items():
            print(f"\n{cat}")
            for k in keys:
                s = P.REGISTRY[k]
                print(f"  {k:22s} {s.title}  [sizes {', '.join(map(str, s.sizes))}; truths {', '.join(s.truths)}]")
        print("\nMethods")
        for g, keys in M.groups().items():
            print(f"  {g}: {', '.join(keys)}")
        return 0
    p, d = _setup(a)
    print(f"{p.title}: n = {p.n}, m = {p.m}, {d.kind} noise {d.level:g}, delta = {d.delta:.3g}"
          + (" (estimated from the data)" if p.meta.get("sigma_estimated") else ""))
    keys = [a.method] if a.cmd == "solve" else (
        [k for k, why in M.applicable(p).items() if not why] if a.method == "all" else a.method.split(","))
    results = []
    for mk in keys:
        if mk not in M.METHODS:
            raise SystemExit(f"unknown method {mk!r}")
        why = M.METHODS[mk].applies(p)
        if why:
            print(f"  skip {mk}: {why}")
            continue
        params = _params(mk, a.set, strict=a.cmd == "solve")
        rule = None
        if a.lam is not None:
            try:
                params["lam"] = float(a.lam)
            except ValueError:
                rule = a.lam
        if rule and "lam" not in M.METHODS[mk].defaults():
            rule = None
        r = _solve(p, d, mk, params, rule)
        results.append(r)
        print("  " + _report(r))
    if a.out and results:
        _save(p, d, results, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
