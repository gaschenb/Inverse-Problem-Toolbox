"""Static matplotlib figures (CLI output and the documentation gallery)."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

INK, MUTED, ACCENT = "#1f2328", "#667085", "#2563eb"
COLORS = ["#2563eb", "#e8590c", "#0f9d58", "#c2255c", "#7048e8", "#0b7285", "#d9480f", "#5c940d", "#862e9c"]


def style(ax, title=None, image=False):
    if image:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, lw=0.4, color="#e6e8eb")
    if title:
        ax.set_title(title, loc="left", fontsize=9.5)


def figure_results(p, d, results, labels=None, ncols=None):
    labels = labels or [r.method for r in results]
    if p.is_image:
        k = len(results) + 2
        ncols = ncols or min(k, 4)
        nrows = int(np.ceil(k / ncols))
        fig = Figure(figsize=(3.0 * ncols, 3.15 * nrows), layout="constrained")
        xt = p.x_true.reshape(p.shape)
        if p.has_truth:
            lo, hi = xt.min(), xt.max()
        else:                                    # measured data: share the range of the reconstructions
            xs = np.concatenate([np.ravel(r.x) for r in results]) if results else np.zeros(1)
            lo, hi = np.percentile(xs, [0.5, 99.5]) if xs.size else (0.0, 1.0)
        axs = [fig.add_subplot(nrows, ncols, i + 1) for i in range(k)]
        axs[0].imshow(xt, cmap="gray", vmin=lo, vmax=hi)
        style(axs[0], "ground truth" if p.has_truth else "ground truth: unknown", True)
        dimg = p.data_image(d.b)
        import matplotlib
        dcm = matplotlib.colormaps["magma" if p.data_kind in ("kspace", "sinogram") else "gray"].with_extremes(bad="#d0d4da")
        axs[1].imshow(dimg, cmap=dcm,
                      aspect="auto" if p.data_kind == "sinogram" else "equal")
        style(axs[1], "data", True)
        for ax, r, lab in zip(axs[2:], results, labels):
            ax.imshow(p.align(r.x).reshape(p.shape), cmap="gray", vmin=lo, vmax=hi)
            if p.has_truth:
                ttl = f"{lab}\nerr {r.info['rel_err']:.3f} · PSNR {r.info['psnr']:.1f}"
            else:
                ttl = f"{lab}\n‖r‖/δ {r.info['discrepancy']:.2f}"
            if p.has_truth and np.isfinite(r.info.get("ssim", np.nan)):
                ttl += f" · SSIM {r.info['ssim']:.2f}"
            style(ax, ttl, True)
        for ax in axs[k:]:
            ax.axis("off")
        return fig
    fig = Figure(figsize=(10, 4.2), layout="constrained")
    ax = fig.add_subplot(111)
    t = (np.arange(p.n) + 0.5) / p.n
    if p.has_truth:
        ax.plot(t, p.x_true, "--", color=INK, lw=1.3, label="truth")
    for r, lab, c in zip(results, labels, COLORS * 3):
        x = p.align(r.x)
        tag = f"{r.info['rel_err']:.3f}" if p.has_truth else f"‖r‖/δ {r.info['discrepancy']:.2f}"
        ax.plot(t, x, color=c, lw=1.4, label=f"{lab} ({tag})")
        if r.std is not None:
            ax.fill_between(t, x - 1.96 * r.std, x + 1.96 * r.std, color=c, alpha=0.15, lw=0)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    style(ax, p.title)
    ax.set_xlabel(p.x_label)
    return fig
