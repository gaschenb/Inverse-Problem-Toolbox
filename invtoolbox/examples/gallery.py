"""Regenerate every figure in the README.

    python -m invtoolbox.examples.gallery docs/figures [--no-gui]
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

from invtoolbox.core import methods as M
from invtoolbox.core import paramchoice as PC
from invtoolbox.core import problems as P
from invtoolbox.core.plots import ACCENT, COLORS, INK, MUTED, figure_results, style

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "mathtext.fontset": "cm",
                            "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130)
    print("wrote", path)


def problems_overview(out):
    keys = list(P.REGISTRY)
    ncol = 5
    nrow = int(np.ceil(len(keys) / ncol))
    fig = Figure(figsize=(15, 3.1 * nrow), layout="constrained")
    sub = fig.subfigures(nrow, ncol)
    for k, sf in zip(keys, sub.ravel()):
        p = P.build(k)
        d = p.simulate(seed=0)
        sf.suptitle(p.title, fontsize=9.5, x=0.02, ha="left")
        a1, a2 = sf.subplots(1, 2)
        if p.is_image:
            a1.imshow(p.x_true.reshape(p.shape), cmap="gray")
            img = p.data_image(d.b)
            a2.imshow(img, cmap="magma" if p.data_kind in ("kspace", "sinogram") else "gray",
                      aspect="auto" if p.data_kind == "sinogram" else "equal")
            style(a1, "x", True)
            style(a2, "b", True)
        else:
            t = (np.arange(p.n) + .5) / p.n
            a1.plot(t, p.x_true, color=INK, lw=1)
            a2.plot(d.b, ".", ms=1.5, color=ACCENT)
            for a, tt in ((a1, "x"), (a2, "b")):
                style(a, tt)
                a.set_xticks([])
                a.tick_params(labelsize=7)
    for sf in sub.ravel()[len(keys):]:
        sf.set_visible(False)
    save(fig, out, "problems.png")


def compare(out, key, cands, name, ncols=None, noise=None, sweep_rule="oracle"):
    p = P.build(key)
    d = p.simulate(*(noise or (None, None)), seed=0)
    res, labels = [], []
    for mk, kw, lab in cands:
        if "lam" in M.METHODS[mk].defaults() and "lam" not in kw and mk != "richardson-lucy":
            sw = PC.sweep(mk, p, d, kw)
            kw = dict(kw, lam=float(sw["lam"][sw["choices"][sweep_rule]]))
        r = M.run(mk, p, d, kw)
        res.append(r)
        labels.append(lab)
        print(f"  {key:18s} {lab:28s} {r.info['rel_err']:.4f}")
    fig = figure_results(p, d, res, labels, ncols=ncols)
    save(fig, out, name)
    return res


def semiconvergence(out):
    p = P.build("ct-limited-angle")
    d = p.simulate(seed=0)
    fig = Figure(figsize=(11, 3.6), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    for (mk, kw, lab), c in zip([("cgls", {"iters": 200}, "CGLS"), ("landweber", {"iters": 1500}, "Landweber"),
                                 ("sirt", {"iters": 1500, "nonneg": False}, "SIRT"),
                                 ("sirt", {"iters": 1500, "nonneg": True}, "SIRT, x ≥ 0")], COLORS):
        r = M.run(mk, p, d, kw)
        k = np.arange(1, len(r.hist["err"]) + 1)
        a1.semilogx(k, r.hist["err"], color=c, lw=1.5, label=lab)
        a2.loglog(k, r.hist["res"], color=c, lw=1.5, label=lab)
        ch = PC.iteration_choice(r.hist, d.delta)
        if "discrepancy" in ch:
            i = ch["discrepancy"]
            a1.plot(k[i], r.hist["err"][i], "s", mfc="none", mec=c, ms=7)
    a2.axhline(1.02 * d.delta, color=MUTED, ls="--", lw=1, label=r"$\tau\delta$")
    style(a1, "Limited-angle CT: relative error vs iteration (squares: discrepancy stop)")
    style(a2, r"Residual $\|Ax_k - b\|$")
    a1.set_xlabel("iteration k")
    a2.set_xlabel("iteration k")
    a1.legend(frameon=False)
    a2.legend(frameon=False)
    save(fig, out, "semiconvergence.png")


def lcurve(out):
    p = P.build("gravity")
    d = p.simulate(seed=0)
    sw = PC.sweep("tikhonov", p, d, {"penalty": "identity"}, np.logspace(-7, 0, 29))
    sp = sw["spectral"]
    fig = Figure(figsize=(11, 3.8), layout="constrained")
    a1, a2, a3 = fig.subplots(1, 3)
    a1.loglog(sp["res"], sp["reg"], color=ACCENT, lw=1.6)
    mk = {"discrepancy": "s", "l-curve": "D", "gcv": "^", "quasi-opt": "v", "oracle": "*"}
    for r, i in sw["choices"].items():
        a1.plot(sw["res"][i], sw["reg"][i], mk[r], ms=9, mfc="none", mec=INK, label=f"{r}: λ={sw['lam'][i]:.1e}")
        a2.semilogx(sw["lam"][i], sw["err"][i], mk[r], ms=9, mfc="none", mec=INK)
    a1.axvline(1.02 * d.delta, color=MUTED, ls="--", lw=1)
    a1.set_xlabel(r"$\|Ax_\lambda - b\|$")
    a1.set_ylabel(r"$\|x_\lambda\|$")
    a1.legend(frameon=False, fontsize=7.5)
    style(a1, "Gravity: L-curve and parameter rules")
    a2.semilogx(sp["lam"], sp["err"], color=ACCENT, lw=1.6, label="relative error")
    a2b = a2.twinx()
    a2b.semilogx(sp["lam"], sp["gcv"] / sp["gcv"].min(), color=MUTED, ls=":", label="GCV")
    a2b.set_yscale("log")
    a2b.spines["right"].set_visible(True)
    style(a2, "Error and GCV vs λ")
    a2.set_xlabel(r"$\lambda$")
    S = __import__("invtoolbox.core.spectral", fromlist=["Spectral"]).Spectral(p)
    s, beta = S.sorted(d.b)
    i = np.arange(1, s.size + 1)
    a3.semilogy(i, s, color=INK, lw=1.3, label=r"$\sigma_i$")
    a3.semilogy(i, beta, ".", ms=3, color=ACCENT, label=r"$|u_i^Tb|$")
    a3.semilogy(i, beta / s, ".", ms=3, color=COLORS[1], label=r"$|u_i^Tb|/\sigma_i$")
    a3.axhline(d.delta / np.sqrt(p.m), color=MUTED, ls="--", lw=1)
    a3.set_ylim(1e-17, 1e3)
    a3.legend(frameon=False)
    style(a3, "Picard plot")
    save(fig, out, "lcurve_picard.png")


def uq(out):
    fig = Figure(figsize=(13, 3.6), layout="constrained")
    a1, a2, a3 = fig.subplots(1, 3, width_ratios=[1.2, 1.2, 0.9])
    p = P.build("backward-heat")
    d = p.simulate(seed=0)
    r = M.run("gaussian-posterior", p, d, {"penalty": "gradient"})
    t = (np.arange(p.n) + .5) / p.n
    a1.fill_between(t, r.lo, r.hi, color=ACCENT, alpha=.2, lw=0, label="95% credible band")
    a1.plot(t, p.x_true, "--", color=INK, lw=1.2, label="truth")
    a1.plot(t, r.x, color=ACCENT, lw=1.5, label="posterior mean")
    a1.legend(frameon=False, fontsize=7.5)
    style(a1, "Backward heat: exact Gaussian posterior")
    p = P.build("coefficient-id")
    d = p.simulate(seed=0)
    r = M.run("gauss-newton", p, d)
    t = (np.arange(p.n) + .5) / p.n
    a2.fill_between(t, r.lo, r.hi, color=COLORS[2], alpha=.2, lw=0, label="Laplace 95% band")
    a2.plot(t, p.x_true, "--", color=INK, lw=1.2, label="truth log a(t)")
    a2.plot(t, r.x, color=COLORS[2], lw=1.5, label="Gauss-Newton MAP")
    a2.legend(frameon=False, fontsize=7.5)
    style(a2, "Coefficient identification: Laplace approximation")
    p = P.build("deblur-motion")
    d = p.simulate(seed=0)
    r = M.run("myula-tv", p, d, {"iters": 3000})
    im = a3.imshow(r.std.reshape(p.shape), cmap="viridis")
    fig.colorbar(im, ax=a3, fraction=.046)
    style(a3, "Motion deblur, TV prior: MYULA posterior std", True)
    save(fig, out, "uq.png")


def phase(out):
    p = P.build("phase-retrieval")
    d = p.simulate(seed=0)
    fig = Figure(figsize=(11, 3.4), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    t = (np.arange(p.n) + .5) / p.n
    a1.plot(t, p.x_true, "--", color=INK, lw=1.2, label="truth")
    a1.plot(t, p.align(M.initial_guess(p, d)), color=MUTED, lw=1, label="spectral init")
    for (mk, lab), c in zip([("wirtinger", "Wirtinger flow"), ("amplitude-flow", "Amplitude flow"),
                             ("gauss-newton", "Gauss-Newton")], COLORS):
        r = M.run(mk, p, d)
        a2.semilogy(np.arange(1, len(r.hist["err"]) + 1), r.hist["err"], color=c, lw=1.5, label=lab)
        if mk == "amplitude-flow":
            a1.plot(t, p.align(r.x), color=c, lw=1.5, label=lab)
    a1.legend(frameon=False)
    a2.legend(frameon=False)
    a2.set_xscale("log")
    a2.set_xlabel("iteration")
    style(a1, "Phase retrieval (m = 6n), error modulo sign")
    style(a2, "Relative error vs iteration")
    save(fig, out, "phase.png")


def gui_shots(out):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["INVTOOLBOX_NO_EXEC"] = "1"
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from invtoolbox.gui.app import MainWindow
    w = MainWindow("light")
    w.resize(1560, 940)
    w.show()

    def wait():
        t = time.time()
        while time.time() - t < 120:
            app.processEvents()
            if not w.busy():
                break
            time.sleep(0.01)
        for _ in range(30):
            app.processEvents()
            time.sleep(0.005)

    def shot(name):
        wait()
        w.grab().save(os.path.join(out, name))
        print("wrote", name)

    wait()
    w.load_problem("deblur-motion"); w.select_method("admm-tv"); w.tabs.setCurrentWidget(w.recon); shot("gui_main.png")
    w.load_problem("gravity"); w.select_method("tikhonov"); wait()
    w.run_sweep(21, 3.5); wait(); shot("gui_lambda.png")
    w.load_problem("shaw"); w.select_method("tsvd"); wait(); w.tabs.setCurrentWidget(w.spec); shot("gui_spectrum.png")
    w.load_problem("backward-heat"); w.select_method("gaussian-posterior", {"penalty": "gradient"}); wait()
    w.tabs.setCurrentWidget(w.uq); shot("gui_uq.png")
    w.load_problem("ct-sparse-view")
    for mk, kw in (("fbp", {}), ("cgls", {"discrepancy": True}), ("sirt", {}), ("pdhg", {"reg": "TV"}), ("pdhg", {"reg": "TGV2"})):
        w.select_method(mk, kw); wait(); w.pin_current()
    w.tabs.setCurrentWidget(w.cmp); shot("gui_compare.png")
    # your own measurements: blurred data saved to disk in physical units, then loaded back (no truth)
    import tempfile
    import numpy as np
    from invtoolbox.core import problems as P
    w.load_problem("deblur-motion")
    p = P.build("deblur-motion", int(w.cb_size.currentText()), w.cb_truth.currentText())
    d = p.simulate("gaussian", 0.01, seed=7)
    f = os.path.join(tempfile.mkdtemp(), "my_blurred_text.npy")
    np.save(f, d.b.reshape(p.data_shape) * p.meta["A_scale"])
    w.load_user_measured(f)
    w.select_method("admm-tv"); w.tabs.setCurrentWidget(w.recon); shot("gui_userdata.png")
    w.clear_user_data()
    w.toggle_theme()
    w.load_problem("mri-radial"); w.select_method("pdhg", {"reg": "TV"}); w.tabs.setCurrentWidget(w.recon); shot("gui_dark.png")
    w.tabs.setCurrentWidget(w.conv); shot("gui_convergence_dark.png")
    w.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv and not argv[0].startswith("-") else os.path.join("docs", "figures")
    os.makedirs(out, exist_ok=True)
    problems_overview(out)
    compare(out, "ct-sparse-view", [("fbp", {}, "FBP (Hann)"), ("cgls", {"discrepancy": True}, "CGLS, discrepancy stop"),
                                    ("sirt", {"iters": 300}, "SIRT, x ≥ 0"), ("tikhonov", {"penalty": "gradient"}, "Tikhonov ‖∇x‖²"),
                                    ("pdhg", {"reg": "TV"}, "TV (PDHG), x ≥ 0"), ("pdhg", {"reg": "TGV2"}, "TGV² (PDHG)")],
            "ct_methods.png", ncols=4)
    compare(out, "deblur-poisson", [("tikhonov", {"penalty": "gradient"}, "Tikhonov ‖∇x‖²"),
                                    ("richardson-lucy", {"iters": 60}, "Richardson-Lucy (60 it)"),
                                    ("pdhg", {"data_term": "L2", "reg": "TV"}, "L2 + TV"),
                                    ("pdhg", {"data_term": "KL", "reg": "TV"}, "KL + TV")], "poisson.png", ncols=3)
    compare(out, "mri-radial", [("adjoint", {}, "zero-filled Aᵀb"), ("fista", {"basis": "db4"}, "FISTA, db4 wavelets"),
                                ("pdhg", {"reg": "TV"}, "TV (PDHG)"), ("pnp-admm", {}, "PnP-ADMM, wavelet denoiser")],
            "mri.png", ncols=3)
    compare(out, "compressed-sensing", [("tikhonov", {}, "ℓ² (Tikhonov)"), ("fista", {}, "ℓ¹ (FISTA)"),
                                        ("omp", {}, "OMP"), ("irls", {}, "IRLS, p = 0.5")], "cs.png")
    compare(out, "deblur-motion", [("tikhonov", {"penalty": "identity"}, "Tikhonov"), ("cgls", {"discrepancy": True}, "CGLS"),
                                   ("fista", {"basis": "db2"}, "FISTA, db2 wavelets"), ("admm-tv", {}, "TV (ADMM)")],
            "motion.png", ncols=3)
    semiconvergence(out)
    lcurve(out)
    uq(out)
    phase(out)
    if "--no-gui" not in argv:
        gui_shots(out)


if __name__ == "__main__":
    main()
