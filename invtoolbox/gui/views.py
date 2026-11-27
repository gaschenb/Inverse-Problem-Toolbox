"""Plot panels. Each view creates its artists once per problem and updates them in place
(set_data / set_ydata), so live previews during iterations cost a single canvas redraw."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from invtoolbox.core.metrics import summary
from invtoolbox.core.spectral import Spectral

from .widgets import THEMES, fmt, muted


class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._relayout = QtCore.QTimer(self)
        self._relayout.setSingleShot(True)
        self._relayout.setInterval(120)
        self._relayout.timeout.connect(self.freeze_layout)

    def reset(self):
        import matplotlib
        self.fig.clear()
        self.fig.set_facecolor(matplotlib.rcParams["figure.facecolor"])
        self.fig.set_layout_engine("constrained")
        self.fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.06)
        return self.fig

    def freeze_layout(self):
        """Run constrained layout once, then pin the positions (fast redraws afterwards)."""
        w, h = self.fig.get_size_inches() * self.fig.dpi
        if w < 120 or h < 120 or not self.fig.axes:
            self.draw_idle()
            return
        try:
            self.fig.set_layout_engine("constrained")
            self.fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.06)
            self.draw()
        except Exception:  # noqa: BLE001  (degenerate sizes during start-up)
            pass
        finally:
            self.fig.set_layout_engine("none")
        self.draw_idle()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._relayout.start()


def _clamp_interval(lo, hi, blo, bhi, log=False):
    """Fit the view [lo, hi] inside the bounds [blo, bhi] (either may be inverted): never wider than the
    bounds, shifted back inside when panned past an edge. Log axes are handled in log space."""
    f = (lambda v: np.log10(max(v, 1e-300))) if log else (lambda v: v)
    g = (lambda v: 10.0 ** v) if log else (lambda v: v)
    lo, hi, blo, bhi = f(lo), f(hi), f(blo), f(bhi)
    flip = (hi < lo) != (bhi < blo)
    if flip:
        lo, hi = hi, lo
    a, b = min(blo, bhi), max(blo, bhi)
    w = abs(hi - lo)
    if w >= b - a:
        lo2, hi2 = (blo, bhi)
    else:
        s = 1 if hi >= lo else -1
        x0 = min(lo, hi)
        x0 = min(max(x0, a), b - w)
        lo2, hi2 = (x0, x0 + w) if s > 0 else (x0 + w, x0)
        if flip:
            lo2, hi2 = hi2, lo2
    return g(lo2), g(hi2)


class NavToolbar(NavigationToolbar2QT):
    """Matplotlib toolbar with a zoom-out limit: pan and zoom never go past the full data extent.

    The extent of each axes is recorded the first time the user pans or zooms it (that is the view the
    plot was drawn with, which covers all of its data) and every later view is clamped to it."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save")]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        import weakref
        self._bounds = weakref.WeakKeyDictionary()

    def _record(self):
        for ax in self.canvas.figure.axes:
            if ax not in self._bounds:
                self._bounds[ax] = (ax.get_xlim(), ax.get_ylim())

    def clamp(self):
        changed = False
        for ax in self.canvas.figure.axes:
            bd = self._bounds.get(ax)
            if bd is None:
                continue
            (bx0, bx1), (by0, by1) = bd
            x = _clamp_interval(*ax.get_xlim(), bx0, bx1, ax.get_xscale() == "log")
            y = _clamp_interval(*ax.get_ylim(), by0, by1, ax.get_yscale() == "log")
            if not np.allclose(x, ax.get_xlim()) or not np.allclose(y, ax.get_ylim()):
                ax.set_xlim(*x, emit=False)
                ax.set_ylim(*y, emit=False)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def forget(self):
        """Call when a plot is redrawn with new data, so its new extent becomes the bound."""
        self._bounds.clear()

    def press_pan(self, event):
        self._record()
        super().press_pan(event)

    def press_zoom(self, event):
        self._record()
        super().press_zoom(event)

    def drag_pan(self, event):
        super().drag_pan(event)
        self.clamp()

    def release_pan(self, event):
        super().release_pan(event)
        self.clamp()

    def release_zoom(self, event):
        super().release_zoom(event)
        self.clamp()


class View(QtWidgets.QWidget):
    def __init__(self, parent=None, toolbar=True):
        super().__init__(parent)
        self.canvas = Canvas(self)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self.lay.setSpacing(4)
        self.top = QtWidgets.QHBoxLayout()
        self.lay.addLayout(self.top)
        self.lay.addWidget(self.canvas, 1)
        self.info = muted("")
        self.info.hide()
        self.lay.insertWidget(1, self.info)
        if toolbar:
            self.nav = NavToolbar(self.canvas, self)
            self.nav.setIconSize(QtCore.QSize(16, 16))
            self.top.addStretch(1)
            self.top.addWidget(self.nav)
        self.theme = THEMES["light"]
        self.dirty = True

    def set_theme(self, t):
        self.theme = t

    def set_info(self, text):
        self.info.setText(text)
        self.info.setVisible(bool(text))

    @property
    def fig(self):
        return self.canvas.fig


def _value_range(problem, data):
    """Display range: the truth when known, otherwise the data (image-like data) or [0, 1]."""
    xt = problem.x_true
    if problem.has_truth:
        return float(xt.min()), float(xt.max())
    if problem.data_kind in ("image", "masked", "lowres"):
        b = np.asarray(data.b, float)
        b = b[np.isfinite(b)]
        if b.size:
            lo, hi = np.percentile(b, [0.5, 99.5])
            if hi > lo:
                return float(lo), float(hi)
    return 0.0, 1.0


def _style_image_axes(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5)


def _nan_cmap(name, bad):
    import matplotlib
    return matplotlib.colormaps[name].with_extremes(bad=bad)


# ============================================================================= reconstruction
class ReconView(View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cb_cmap = QtWidgets.QComboBox()
        self.cb_cmap.addItems(["gray", "viridis", "magma", "inferno", "cividis", "bone"])
        self.cb_cmap.currentTextChanged.connect(self._recolor)
        self.top.insertWidget(0, QtWidgets.QLabel("Colormap"))
        self.top.insertWidget(1, self.cb_cmap)
        self.chips = []
        for _ in range(5):
            c = QtWidgets.QLabel("")
            c.setObjectName("chip")
            c.hide()
            self.top.insertWidget(2 + len(self.chips), c)
            self.chips.append(c)
        self.canvas.setToolTip("Click an image to move the profile line")
        self.problem = None
        self.row = 0
        self.canvas.mpl_connect("button_press_event", self._click)
        self._band = None

    # ------------------------------------------------------------------ setup
    def setup(self, problem, data):
        self.problem, self.data = problem, data
        self.canvas.reset()
        t = self.theme
        if problem.is_image:
            N = problem.shape
            gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1])
            self.ax_t = self.fig.add_subplot(gs[0, 0])
            self.ax_d = self.fig.add_subplot(gs[0, 1])
            self.ax_r = self.fig.add_subplot(gs[0, 2])
            self.ax_e = self.fig.add_subplot(gs[1, 0])
            self.ax_p = self.fig.add_subplot(gs[1, 1:])
            xt = problem.x_true.reshape(N)
            self.vmin, self.vmax = _value_range(problem, data)
            cm = _nan_cmap(self.cb_cmap.currentText(), t["bad"])
            kw = dict(cmap=cm, vmin=self.vmin, vmax=self.vmax, interpolation="nearest")
            self.im_t = self.ax_t.imshow(xt, **kw)
            dimg = problem.data_image(data.b)
            aspect = "auto" if problem.data_kind == "sinogram" else "equal"
            dcm = _nan_cmap("magma" if problem.data_kind in ("kspace", "sinogram") else self.cb_cmap.currentText(),
                            t["bad"])
            self.im_d = self.ax_d.imshow(dimg, cmap=dcm, interpolation="nearest", aspect=aspect)
            self.im_r = self.ax_r.imshow(np.zeros(N), **kw)
            self.im_e = self.ax_e.imshow(np.zeros(N), cmap="magma", vmin=0, vmax=max(self.vmax - self.vmin, 1e-12) * 0.5,
                                         interpolation="nearest")
            _style_image_axes(self.ax_t, "Ground truth" if problem.has_truth else "Ground truth: unknown (measured data)")
            _style_image_axes(self.ax_d, {"sinogram": "Sinogram b", "kspace": "k-space  log|b|",
                                          "masked": "Observed pixels", "lowres": "Low-res data b"}.get(problem.data_kind, "Data b"))
            _style_image_axes(self.ax_r, "Reconstruction")
            _style_image_axes(self.ax_e, "|error|" if problem.has_truth else "|error|: needs a ground truth")
            self.fig.colorbar(self.im_r, ax=self.ax_r, fraction=0.046, pad=0.02).outline.set_visible(False)
            self.fig.colorbar(self.im_e, ax=self.ax_e, fraction=0.046, pad=0.02).outline.set_visible(False)
            self.row = N[0] // 2
            self.hl = [ax.axhline(self.row, color=t["accent"], lw=0.8, alpha=0.8) for ax in (self.ax_t, self.ax_r, self.ax_e)]
            xs = np.arange(N[1])
            (self.pl_t,) = self.ax_p.plot(xs, xt[self.row], color=t["plot_fg"], lw=1.2, ls="--", label="truth")
            (self.pl_r,) = self.ax_p.plot(xs, np.zeros(N[1]), color=t["accent"], lw=1.6, label="reconstruction")
            self.ax_p.set_xlim(0, N[1] - 1)
            pad = 0.1 * (self.vmax - self.vmin + 1e-12)
            self.ax_p.set_ylim(self.vmin - pad, self.vmax + pad)
            self.ax_p.set_title(f"Profile, row {self.row}", loc="left", fontsize=9.5)
            self.ax_p.legend(loc="upper right")
            self.ax_p.grid(True, lw=0.5)
        else:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[1.6, 1])
            self.ax_x = self.fig.add_subplot(gs[0])
            self.ax_b = self.fig.add_subplot(gs[1])
            xt = problem.x_true
            tgrid = np.arange(problem.n) if problem.x_label == "index" else (np.arange(problem.n) + 0.5) / problem.n
            self.tgrid = tgrid
            (self.l_t,) = self.ax_x.plot(tgrid, xt, color=t["plot_fg"], lw=1.3, ls="--", label="truth")
            (self.l_r,) = self.ax_x.plot(tgrid, np.zeros_like(xt), color=t["accent"], lw=1.8, label="reconstruction")
            self._band = None
            lo, hi = _value_range(problem, data)
            pad = 0.25 * (hi - lo + 1e-12)
            self.ax_x.set_ylim(lo - pad, hi + pad)
            self.ax_x.set_xlim(tgrid[0], tgrid[-1])
            self.ax_x.set_title("Solution" if problem.has_truth else "Solution (no ground truth: measured data)",
                                loc="left", fontsize=9.5)
            self.ax_x.set_xlabel(problem.x_label)
            self.ax_x.legend(loc="upper right", ncol=3)
            self.ax_x.grid(True, lw=0.5)
            b = data.b
            ib = np.arange(b.size)
            self.ax_b.plot(ib, b, ".", ms=2.5, color=t["muted"], label="data b")
            (self.l_p,) = self.ax_b.plot(ib, np.zeros_like(b), color=t["accent"], lw=1.3, label=r"$F(\hat{x})$")
            if problem.data_kind == "multi":
                ne = problem.meta["n_exp"]
                for k in range(1, ne):
                    self.ax_b.axvline(k * b.size / ne, color=t["border"], lw=1)
            self.ax_b.set_xlim(0, b.size - 1)
            self.ax_b.set_title("Data and prediction", loc="left", fontsize=9.5)
            self.ax_b.legend(loc="upper right", ncol=2)
            self.ax_b.grid(True, lw=0.5)
        self.canvas.freeze_layout()

    def set_data(self, data):
        self.data = data
        if self.problem is None:
            return
        if self.problem.is_image:
            dimg = self.problem.data_image(data.b)
            self.im_d.set_data(dimg)
            finite = dimg[np.isfinite(dimg)]
            if finite.size:
                self.im_d.set_clim(np.percentile(finite, 0.5), np.percentile(finite, 99.5))
        else:
            self.setup(self.problem, data)
            return
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ updates
    def update_x(self, x, std=None):
        p = self.problem
        if p is None or x is None:
            return
        x = p.align(np.asarray(x))
        if p.is_image:
            img = x.reshape(p.shape)
            self.im_r.set_data(img)
            self.im_e.set_data(np.abs(img - p.x_true.reshape(p.shape)))
            self.pl_r.set_ydata(img[self.row])
            if not p.has_truth:
                fin = img[np.isfinite(img)]
                if fin.size:
                    lo, hi = np.percentile(fin, [0.5, 99.5])
                    hi = hi if hi > lo else lo + 1e-12
                    self.im_r.set_clim(lo, hi)
                    pad = 0.1 * (hi - lo)
                    self.ax_p.set_ylim(lo - pad, hi + pad)
        else:
            self.l_r.set_ydata(x)
            if not p.has_truth and np.isfinite(x).any():
                lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
                pad = 0.15 * (hi - lo + 1e-12)
                self.ax_x.set_ylim(lo - pad, hi + pad)
            if self._band is not None:
                self._band.remove()
                self._band = None
            if std is not None:
                self._band = self.ax_x.fill_between(self.tgrid, x - 1.96 * std, x + 1.96 * std, color=self.theme["accent"],
                                                    alpha=0.18, lw=0, label="95% band")
            self.l_p.set_ydata(p.forward(x))
        self.canvas.draw_idle()

    def set_chips(self, items):
        for c, txt in zip(self.chips, list(items) + [None] * len(self.chips)):
            c.setVisible(bool(txt))
            if txt:
                c.setText(txt)

    def _click(self, ev):
        p = self.problem
        if p is None or not p.is_image or ev.inaxes not in (self.ax_t, self.ax_r, self.ax_e) or ev.ydata is None:
            return
        if self.nav.mode:
            return
        self.row = int(np.clip(round(ev.ydata), 0, p.shape[0] - 1))
        for h in self.hl:
            h.set_ydata([self.row, self.row])
        self.pl_t.set_ydata(p.x_true.reshape(p.shape)[self.row])
        self.pl_r.set_ydata(self.im_r.get_array()[self.row])
        self.ax_p.set_title(f"Profile, row {self.row}", loc="left", fontsize=9.5)
        self.canvas.draw_idle()

    def _recolor(self, name):
        if self.problem is None or not self.problem.is_image:
            return
        cm = _nan_cmap(name, self.theme["bad"])
        self.im_t.set_cmap(cm)
        self.im_r.set_cmap(cm)
        if self.problem.data_kind not in ("kspace", "sinogram"):
            self.im_d.set_cmap(cm)
        self.canvas.draw_idle()


# ============================================================================= convergence
class ConvergenceView(View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup()

    def setup(self):
        self.canvas.reset()
        t = self.theme
        self.ax_o = self.fig.add_subplot(131)
        self.ax_e = self.fig.add_subplot(132)
        self.ax_r = self.fig.add_subplot(133)
        for ax, ttl in ((self.ax_o, "Objective"), (self.ax_e, "Relative error"), (self.ax_r, "Residual  ||F(x) - b||")):
            ax.set_title(ttl, loc="left", fontsize=9.5)
            ax.set_xlabel("iteration")
            ax.grid(True, lw=0.5)
        (self.l_o,) = self.ax_o.semilogy([], [], color=t["accent"], lw=1.5)
        (self.l_e,) = self.ax_e.plot([], [], color=t["accent"], lw=1.5)
        (self.l_r,) = self.ax_r.semilogy([], [], color=t["accent"], lw=1.5)
        self.l_delta = self.ax_r.axhline(1, color=t["muted"], ls="--", lw=1, label=r"$\tau\delta$")
        self.m_best = self.ax_e.plot([], [], "o", ms=5, color=t["accent"], mfc="none")[0]
        self.m_disc = self.ax_r.plot([], [], "o", ms=5, color=t["muted"], mfc="none")[0]
        self.ax_r.legend(loc="upper right")
        self.ghosts = []
        self.canvas.freeze_layout()

    def update_hist(self, hist, delta=None, others=()):
        obj = np.asarray(hist.get("obj", []), float)
        err = np.asarray(hist.get("err", []), float)
        res = np.asarray(hist.get("res", []), float)
        k = np.arange(err.size)
        ok = np.isfinite(obj) & (obj > 0)
        self.l_o.set_data(k[ok], obj[ok])
        self.l_e.set_data(k, err)
        self.l_r.set_data(k, res)
        for g in self.ghosts:
            g.remove()
        self.ghosts = []
        for label, h, col in others:
            e = np.asarray(h.get("err", []))
            if e.size > 1:
                self.ghosts += self.ax_e.plot(np.arange(e.size), e, lw=1, alpha=0.7, color=col, label=label)
        if self.ghosts:
            self.ax_e.legend(loc="upper right")
        elif self.ax_e.get_legend():
            self.ax_e.get_legend().remove()
        if delta is not None:
            self.l_delta.set_ydata([1.02 * delta] * 2)
        if err.size:
            i = int(np.argmin(err))
            self.m_best.set_data([i], [err[i]])
            hit = np.flatnonzero(res <= 1.02 * (delta or 0))
            if hit.size:
                self.m_disc.set_data([hit[0]], [res[hit[0]]])
            else:
                self.m_disc.set_data([], [])
            msg = f"{err.size} iterations · min error {err[i]:.4f} at k = {i}"
            if hit.size:
                msg += f" · discrepancy reached at k = {hit[0]}"
            self.set_info(msg)
        for ax in (self.ax_o, self.ax_e, self.ax_r):
            ax.relim()
            ax.autoscale_view()
        if delta is not None and res.size:
            lo = min(res.min(), delta) * 0.7
            hi = max(res.max(), delta) * 1.4
            self.ax_r.set_ylim(lo, hi)
        self.canvas.draw_idle()


# ============================================================================= lambda study
class LambdaView(View):
    apply_lam = QtCore.pyqtSignal(float)
    apply_iters = QtCore.pyqtSignal(int)
    run_sweep = QtCore.pyqtSignal(int, float)

    RULES = ("discrepancy", "l-curve", "gcv", "quasi-opt", "oracle")
    MARK = {"discrepancy": "s", "l-curve": "D", "gcv": "^", "quasi-opt": "v", "oracle": "*"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.btn = QtWidgets.QPushButton("Run λ sweep")
        self.btn.setObjectName("primary")
        self.btn.clicked.connect(lambda: self.run_sweep.emit(self.sp_n.value(), self.sp_dec.value()))
        self.sp_n = QtWidgets.QSpinBox()
        self.sp_n.setRange(5, 61)
        self.sp_n.setValue(15)
        self.sp_dec = QtWidgets.QDoubleSpinBox()
        self.sp_dec.setRange(0.5, 6)
        self.sp_dec.setValue(3.0)
        self.sp_dec.setSingleStep(0.5)
        self.rule_btns = {}
        for w in (self.btn, QtWidgets.QLabel("points"), self.sp_n, QtWidgets.QLabel("± decades"), self.sp_dec):
            self.top.insertWidget(self.top.count() - 2, w)
        self.top.insertSpacing(self.top.count() - 2, 12)
        for r in self.RULES:
            b = QtWidgets.QPushButton("Use " + ("GCV" if r == "gcv" else r))
            b.setEnabled(False)
            b.clicked.connect(lambda _=None, rr=r: self._apply(rr))
            self.top.insertWidget(self.top.count() - 2, b)
            self.rule_btns[r] = b
        self.sw = None
        self.iter_mode = False
        self.canvas.mpl_connect("pick_event", self._pick)
        self.blank("Press “Run λ sweep” to scan the regularization parameter.")

    def blank(self, msg):
        self.canvas.reset()
        self.fig.text(0.5, 0.5, msg, ha="center", va="center", color=self.theme["muted"], fontsize=10)
        self.sw = None
        for b in self.rule_btns.values():
            b.setEnabled(False)
        self.canvas.draw_idle()

    def set_mode(self, has_lam, iterative):
        self.iter_mode = (not has_lam) and iterative
        self.btn.setEnabled(has_lam)
        self.sp_n.setEnabled(has_lam)
        self.sp_dec.setEnabled(has_lam)
        if not has_lam and not iterative:
            self.blank("This method has no regularization parameter to scan.")

    def show_iterations(self, hist, delta):
        """Semiconvergence view for iterative regularization (k is the parameter)."""
        self.canvas.reset()
        t = self.theme
        err, res = np.asarray(hist["err"]), np.asarray(hist["res"])
        if err.size < 2:
            self.blank("Run the iterative method to see its semiconvergence.")
            return
        ax = self.fig.add_subplot(121)
        k = np.arange(1, err.size + 1)
        ax.semilogx(k, err, color=t["accent"], lw=1.6)
        ax.set_title("Semiconvergence: error vs k", loc="left", fontsize=9.5)
        ax.set_xlabel("iteration k (regularization parameter)")
        ax.grid(True, lw=0.5)
        ax2 = self.fig.add_subplot(122)
        ax2.loglog(k, res, color=t["accent"], lw=1.6, label="||Ax_k - b||")
        ax2.axhline(1.02 * delta, color=t["muted"], ls="--", label=r"$\tau\delta$")
        ax2.set_title("Residual vs k", loc="left", fontsize=9.5)
        ax2.set_xlabel("iteration k")
        ax2.grid(True, lw=0.5)
        hit = np.flatnonzero(res <= 1.02 * delta)
        io = int(np.argmin(err))
        ax.plot(k[io], err[io], "*", ms=11, color=t["accent"], label=f"oracle k = {k[io]}")
        self._iters = {"oracle": int(k[io])}
        if hit.size:
            ax.plot(k[hit[0]], err[hit[0]], "s", ms=7, color=t["muted"], label=f"discrepancy k = {k[hit[0]]}")
            self._iters["discrepancy"] = int(k[hit[0]])
        ax.legend()
        ax2.legend()
        for r, b in self.rule_btns.items():
            b.setEnabled(r in self._iters)
        self.set_info("Iterative regularization: stopping index k plays the role of λ. "
                          "Use a rule button to set the iteration count.")
        self.canvas.freeze_layout()

    def show_sweep(self, sw, delta, cur_lam=None):
        self.sw = sw
        self.canvas.reset()
        t = self.theme
        lam, res, reg, err = sw["lam"], sw["res"], sw["reg"], sw["err"]
        ax = self.fig.add_subplot(121)
        ax.loglog(res, reg, "-", color=t["border"], lw=1.2, zorder=1)
        sc = ax.scatter(res, reg, c=np.log10(lam), cmap="viridis", s=26, zorder=2, picker=True)
        sc.set_picker(6)
        self._sc = sc
        ax.axvline(1.02 * delta, color=t["muted"], ls="--", lw=1)
        ax.set_xlabel(r"$\|A x_\lambda - b\|$")
        ax.set_ylabel(r"$\mathcal{R}(x_\lambda)$")
        ax.set_title("L-curve (click a point to use its λ)", loc="left", fontsize=9.5)
        ax.grid(True, which="both", lw=0.4)
        cb = self.fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$\log_{10}\lambda$")
        cb.outline.set_visible(False)
        ax2 = self.fig.add_subplot(122)
        ax2.semilogx(lam, err, "-o", ms=3, color=t["accent"], lw=1.6, label="relative error")
        if "spectral" in sw:
            sp = sw["spectral"]
            ax2.semilogx(sp["lam"], sp["err"], color=t["accent"], lw=0.8, alpha=0.5)
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel("relative error", color=t["accent"])
        ax2.grid(True, which="both", lw=0.4)
        ax3 = ax2.twinx()
        ax3.semilogx(lam, res / delta, color=t["muted"], lw=1.2, ls="-", label=r"$\|r\| / \delta$")
        if "gcv" in sw:
            g = np.asarray(sw["gcv"])
            ax3.semilogx(lam, g / g.min(), color=t["muted"], lw=1, ls=":", label="GCV (scaled)")
        ax3.axhline(1.02, color=t["muted"], ls="--", lw=0.8)
        ax3.set_yscale("log")
        ax3.spines["right"].set_visible(True)
        ax3.set_ylabel(r"$\|r\|/\delta$", color=t["muted"])
        ax2.set_title("Error and discrepancy vs λ", loc="left", fontsize=9.5)
        lines = []
        txt = []
        for r, i in sw["choices"].items():
            mk = self.MARK.get(r, "o")
            lines.append(ax.plot(res[i], reg[i], mk, ms=9, mfc="none", mec=t["plot_fg"], mew=1.3, label=r)[0])
            ax2.plot(lam[i], err[i], mk, ms=9, mfc="none", mec=t["plot_fg"], mew=1.3)
            txt.append(f"{r}: λ = {fmt(lam[i])} (err {err[i]:.3f})")
        if cur_lam is not None:
            ax2.axvline(cur_lam, color=t["accent"], lw=0.8, alpha=0.6)
        ax.legend(loc="upper right")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="upper center")
        for r, b in self.rule_btns.items():
            b.setEnabled(r in sw["choices"])
        self.set_info("   ·   ".join(txt))
        self.canvas.freeze_layout()

    def _pick(self, ev):
        if self.sw is None or ev.artist is not getattr(self, "_sc", None):
            return
        i = int(ev.ind[0])
        self.apply_lam.emit(float(self.sw["lam"][i]))

    def _apply(self, rule):
        if self.iter_mode:
            self.apply_iters.emit(self._iters[rule])
        elif self.sw is not None:
            self.apply_lam.emit(float(self.sw["lam"][self.sw["choices"][rule]]))


# ============================================================================= spectrum
class SpectrumView(View):
    def __init__(self, parent=None):
        super().__init__(parent)

    def show(self, problem, data, method_key=None, params=None):
        self.canvas.reset()
        t = self.theme
        if not problem.linear or not Spectral.available(problem):
            reason = "nonlinear forward model" if not problem.linear else "matrix-free operator without a cheap SVD"
            self.fig.text(0.5, 0.5, f"Spectral analysis unavailable ({reason}).\n"
                                    "Available for convolution problems (Fourier) and small explicit matrices (SVD).",
                          ha="center", va="center", color=t["muted"], fontsize=10)
            self.set_info("")
            self.canvas.draw_idle()
            return
        S = Spectral(problem)
        s, beta = S.sorted(data.b)
        i = np.arange(1, s.size + 1)
        ax = self.fig.add_subplot(121)
        noise = data.delta / np.sqrt(problem.m)
        many = s.size > 600
        ms = 1.5 if many else 3
        ax.semilogy(i, s, color=t["plot_fg"], lw=1.4, label=r"$\sigma_i$")
        ax.semilogy(i, beta, ".", ms=ms, color=t["accent"], label=r"$|u_i^{T} b|$", rasterized=many)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(s > 1e-13 * s.max(), beta / s, np.nan)   # drop numerically-zero σ (round-off only)
        ax.semilogy(i, q, ".", ms=ms, color=SERIES_ALT, label=r"$|u_i^{T} b| / \sigma_i$", rasterized=many)
        ax.axhline(noise, color=t["muted"], ls="--", lw=1, label=r"noise level $\delta/\sqrt{m}$")
        qf = q[np.isfinite(q)]
        top = max(beta.max(), s.max(), qf.max() if qf.size else 0) * 3
        ax.set_ylim(max(min(s[s > 0].min(), noise) * 0.1, top * 1e-18), top)
        ax.set_xlabel("index i (sorted by σ)" if S.kind == "fourier" else "index i")
        ax.set_title("Picard plot" + (" (Fourier modes)" if S.kind == "fourier" else " (SVD)"), loc="left", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.85, frameon=True)
        ax.grid(True, lw=0.4)
        k_noise = int(np.argmax(beta < noise)) if np.any(beta < noise) else s.size
        cond = s[0] / max(s[-1], 1e-300)
        self.set_info(f"{S.kind.upper()}: {s.size} components · cond(A) = {cond:.2e} · "
                          f"|uᵢᵀb| reaches the noise floor near i ≈ {k_noise}")
        ax2 = self.fig.add_subplot(122)
        lam = (params or {}).get("lam")
        drawn = False
        if method_key in ("tikhonov", "gaussian-posterior") and lam:
            pen = (params or {}).get("penalty", "identity")
            if pen == "identity" or S.kind == "fourier":
                if S.kind == "fourier" and pen != "identity":
                    from invtoolbox.core.methods import _fourier_penalty_symbol
                    G = _fourier_penalty_symbol(S.shape, pen).ravel()
                    o = np.argsort(-S.s)
                    phi = (S.s[o] ** 2) / (S.s[o] ** 2 + lam * G[o])
                else:
                    phi = s ** 2 / (s ** 2 + lam)
                ax2.semilogx(i, phi, ".", ms=ms, color=t["accent"], rasterized=many)
                ax2.set_title(f"Filter factors $\\varphi_i$, Tikhonov λ = {fmt(lam)}", loc="left", fontsize=9.5)
                drawn = True
        if method_key == "tsvd":
            k = (params or {}).get("k_used") or (params or {}).get("k") or 0
            if k:
                phi = (i <= k).astype(float)
                ax2.semilogx(i, phi, color=t["accent"], lw=1.6)
                ax2.set_title(f"Filter factors $\\varphi_i$, TSVD k = {k}", loc="left", fontsize=9.5)
                drawn = True
        if not drawn:
            for lam_ in (1e-6, 1e-4, 1e-2):
                ax2.semilogx(i, s ** 2 / (s ** 2 + lam_), lw=1.2, label=f"Tikhonov λ = {lam_:g}")
            ax2.set_title("Filter factors $\\varphi_i$ (reference)", loc="left", fontsize=9.5)
            ax2.legend(loc="lower left")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("index i")
        ax2.grid(True, lw=0.4)
        self.canvas.freeze_layout()


SERIES_ALT = "#e8590c"


# ============================================================================= uncertainty
class UncertaintyView(View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.show(None, None)

    def show(self, problem, res):
        self.canvas.reset()
        t = self.theme
        if problem is None or res is None or res.std is None:
            self.fig.text(0.5, 0.5, "Run a Bayesian method (Gaussian posterior, MYULA) or Gauss-Newton\n"
                                    "to see posterior standard deviations and credible intervals.",
                          ha="center", va="center", color=t["muted"], fontsize=10)
            self.set_info("")
            self.canvas.draw_idle()
            return
        x = problem.align(res.x)
        std = res.std
        err = np.abs(x - problem.x_true)
        cover = float(np.mean(err <= 1.96 * std + 1e-15))
        z = err / np.maximum(std, 1e-15)
        self.set_info(f"95 % credible-interval coverage of the truth: {100 * cover:.1f} %   ·   "
                          f"mean std {std.mean():.3g}   ·   median |error|/std {np.median(z):.2f}   ·   "
                          f"{res.info.get('uq', res.method)}")
        if problem.is_image:
            sh = problem.shape
            axs = [self.fig.add_subplot(1, 3, k + 1) for k in range(3)]
            xt = problem.x_true
            ims = [axs[0].imshow(x.reshape(sh), cmap="gray", vmin=xt.min(), vmax=xt.max()),
                   axs[1].imshow(std.reshape(sh), cmap="viridis"),
                   axs[2].imshow(np.clip(z, 0, 4).reshape(sh), cmap="magma", vmin=0, vmax=4)]
            for ax, im, ttl in zip(axs, ims, ("Posterior mean", "Posterior std", "|error| / std  (clipped at 4)")):
                _style_image_axes(ax, ttl)
                self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).outline.set_visible(False)
        else:
            tg = (np.arange(problem.n) + 0.5) / problem.n
            ax = self.fig.add_subplot(211)
            ax.fill_between(tg, x - 1.96 * std, x + 1.96 * std, color=t["accent"], alpha=0.2, lw=0, label="95% credible band")
            ax.plot(tg, problem.x_true, "--", color=t["plot_fg"], lw=1.2, label="truth")
            ax.plot(tg, x, color=t["accent"], lw=1.6, label="posterior mean / MAP")
            ax.legend(loc="upper right", ncol=3)
            ax.set_title("Posterior", loc="left", fontsize=9.5)
            ax.grid(True, lw=0.4)
            ax2 = self.fig.add_subplot(212)
            ax2.plot(tg, std, color=t["accent"], lw=1.4, label="posterior std")
            ax2.plot(tg, err, color=t["muted"], lw=1, label="|error|")
            ax2.legend(loc="upper right")
            ax2.set_xlabel(problem.x_label)
            ax2.grid(True, lw=0.4)
        self.canvas.freeze_layout()


# ============================================================================= compare
class CompareView(QtWidgets.QWidget):
    COLS = ["", "method", "settings", "rel. error", "PSNR", "SSIM", "||r||/δ", "iters", "time"]
    rows_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        top = QtWidgets.QHBoxLayout()
        self.lbl = muted("Pinned runs on the current problem. Solve (F5) pins automatically; Ctrl+P pins a live result.")
        top.addWidget(self.lbl, 1)
        self.btn_rm = QtWidgets.QPushButton("Remove selected")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        top.addWidget(self.btn_rm)
        top.addWidget(self.btn_clear)
        lay.addLayout(top)
        split = QtWidgets.QSplitter(Qt.Vertical)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self.redraw)
        split.addWidget(self.table)
        self.canvas = Canvas(self)
        split.addWidget(self.canvas)
        split.setSizes([220, 520])
        lay.addWidget(split, 1)
        self.runs = []
        self.problem = None
        self.theme = THEMES["light"]
        self.btn_clear.clicked.connect(self.clear)
        self.btn_rm.clicked.connect(self.remove_selected)

    def set_theme(self, t):
        self.theme = t

    def set_problem(self, problem):
        if problem is not self.problem:
            self.problem = problem
            self.clear()

    def clear(self):
        self.runs = []
        self.table.setRowCount(0)
        self.redraw()
        self.rows_changed.emit()

    def remove_selected(self):
        ids = {int(self.table.item(i.row(), 0).data(Qt.UserRole)) for i in self.table.selectionModel().selectedRows()}
        self.runs = [r for r in self.runs if r["id"] not in ids]
        self._fill()
        self.rows_changed.emit()

    def add(self, label, settings, res, color):
        rid = (max([r["id"] for r in self.runs]) + 1) if self.runs else 1
        self.runs.append({"id": rid, "label": label, "settings": settings, "res": res, "color": color})
        self.table.clearSelection()
        self._fill()
        self.rows_changed.emit()

    def _fill(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.runs))
        for i, r in enumerate(self.runs):
            info = r["res"].info
            vals = [str(r["id"]), r["label"], r["settings"], info["rel_err"], info["psnr"], info["ssim"],
                    info["discrepancy"], len(r["res"].hist.get("err", [])), r["res"].seconds]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem()
                if isinstance(v, float) and not np.isfinite(v):
                    it.setText("–")
                elif isinstance(v, float):
                    it.setData(Qt.DisplayRole, float(f"{v:.4g}"))
                elif isinstance(v, int):
                    it.setData(Qt.DisplayRole, v)
                else:
                    it.setText(v)
                if j == 0:
                    it.setData(Qt.UserRole, r["id"])
                    it.setForeground(QtGui_color(r["color"]))
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.redraw()

    def selected(self):
        ids = [int(self.table.item(i.row(), 0).data(Qt.UserRole)) for i in self.table.selectionModel().selectedRows()]
        sel = [r for r in self.runs if r["id"] in ids]
        return sel or self.runs[-6:]

    def redraw(self):
        fig = self.canvas.reset()
        t = self.theme
        p = self.problem
        runs = self.selected()[:8]
        if p is None or not runs:
            fig.text(0.5, 0.5, "No pinned runs yet.", ha="center", va="center", color=t["muted"])
            self.canvas.draw_idle()
            return
        if p.is_image:
            k = len(runs) + 1
            cols = min(k, 5)
            rows = int(np.ceil(k / cols))
            gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [1.3])
            xt = p.x_true.reshape(p.shape)
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(xt, cmap="gray", vmin=xt.min(), vmax=xt.max(), interpolation="nearest")
            _style_image_axes(ax, "truth")
            for j, r in enumerate(runs, start=1):
                ax = fig.add_subplot(gs[j // cols, j % cols])
                ax.imshow(p.align(r["res"].x).reshape(p.shape), cmap="gray", vmin=xt.min(), vmax=xt.max(),
                          interpolation="nearest")
                _style_image_axes(ax, f"#{r['id']} {r['label']}\nerr {r['res'].info['rel_err']:.3f} · "
                                      f"PSNR {r['res'].info['psnr']:.1f}")
                ax.title.set_color(r["color"])
            axc = fig.add_subplot(gs[:, cols])
        else:
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
            ax = fig.add_subplot(gs[0])
            tg = (np.arange(p.n) + 0.5) / p.n
            ax.plot(tg, p.x_true, "--", color=t["plot_fg"], lw=1.3, label="truth")
            for r in runs:
                ax.plot(tg, p.align(r["res"].x), color=r["color"], lw=1.4,
                        label=f"#{r['id']} {r['label']} ({r['res'].info['rel_err']:.3f})")
            ax.legend(loc="upper right", fontsize=7.5)
            ax.grid(True, lw=0.4)
            axc = fig.add_subplot(gs[1])
        any_hist = False
        for r in runs:
            e = np.asarray(r["res"].hist.get("err", []))
            if e.size > 1:
                axc.plot(np.arange(1, e.size + 1), e, color=r["color"], lw=1.3, label=f"#{r['id']}")
                any_hist = True
            elif e.size == 1:
                axc.axhline(e[0], color=r["color"], lw=1, ls=":", label=f"#{r['id']} (direct)")
                any_hist = True
        if any_hist:
            axc.set_xscale("log")
            axc.legend(fontsize=7.5)
        axc.set_title("Relative error vs iteration", loc="left", fontsize=9.5)
        axc.grid(True, lw=0.4)
        self.canvas.freeze_layout()


def QtGui_color(hexcol):
    from PyQt5 import QtGui
    return QtGui.QBrush(QtGui.QColor(hexcol))
