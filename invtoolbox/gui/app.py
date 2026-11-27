"""Inverse Problem Toolbox - PyQt5 desktop GUI.

    invtoolbox-gui            or     python -m invtoolbox.gui

Left: problem and method cards (equations rendered in-app from mathtext outlines). Right: tabs for the
reconstruction, convergence, lambda study (L-curve / discrepancy / GCV / oracle), Picard plot,
uncertainty and run comparison. With "Live" on, every edit re-solves in a background thread
(debounced, the previous solve is cancelled) and iterates are streamed into the plots.
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import matplotlib

matplotlib.use("Qt5Agg")

from invtoolbox.core import methods as M
from invtoolbox.core import paramchoice as PC
from invtoolbox.core import problems as P
from invtoolbox.core import userdata as U
from invtoolbox.core.defaults import default_lam, variant

from .views import CompareView, ConvergenceView, LambdaView, ReconView, SpectrumView, UncertaintyView
from .widgets import (SERIES, SERIES_DARK, EquationView, GroupedCombo, ParamForm, Worker, apply_theme, card, fmt,
                      muted)

RECOMMENDED = {
    "deblur-gaussian": ("admm-tv", {}), "deblur-motion": ("admm-tv", {}), "deblur-defocus": ("pdhg", {"reg": "TGV2"}),
    "deblur-poisson": ("pdhg", {"data_term": "KL", "reg": "TV"}), "denoise": ("pdhg", {"reg": "TV"}),
    "inpainting": ("pdhg", {"reg": "TV"}), "super-resolution": ("admm-tv", {}),
    "ct-sparse-view": ("pdhg", {"reg": "TV"}), "ct-limited-angle": ("pdhg", {"reg": "TV"}),
    "mri-cartesian": ("fista", {"basis": "db4"}), "mri-radial": ("pdhg", {"reg": "TV"}),
    "compressed-sensing": ("fista", {}), "spike-deconvolution": ("fista", {}), "gravity": ("tikhonov", {"penalty": "gradient"}),
    "shaw": ("tsvd", {}), "backward-heat": ("admm-tv", {}), "differentiation": ("admm-tv", {}),
    "inverse-laplace": ("tikhonov", {"penalty": "laplacian"}), "phase-retrieval": ("amplitude-flow", {}),
    "coefficient-id": ("gauss-newton", {}), "custom": ("tikhonov", {}),
}

DATA_FILES = "Arrays and images (*.npy *.npz *.csv *.txt *.mat *.mtx *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"

EXAMPLES = [
    ("Gaussian deblurring: TV beats Tikhonov", dict(problem="deblur-gaussian", method="admm-tv")),
    ("Motion blur on text: TV", dict(problem="deblur-motion", method="admm-tv")),
    ("Defocused bar target: TGV", dict(problem="deblur-defocus", method="pdhg", params={"reg": "TGV2"})),
    ("Photon-limited stars: Richardson-Lucy", dict(problem="deblur-poisson", method="richardson-lucy")),
    ("Photon-limited stars: KL + TV", dict(problem="deblur-poisson", method="pdhg", params={"data_term": "KL", "reg": "TV"})),
    ("Salt-and-pepper noise: L1 data + TV", dict(problem="denoise", noise=("salt-pepper", 0.1), method="pdhg",
                                                 params={"data_term": "L1", "reg": "TV"})),
    ("Denoising with a plug-and-play wavelet prior", dict(problem="denoise", method="pnp-admm")),
    ("Inpainting 70 % missing pixels: TV", dict(problem="inpainting", method="pdhg", params={"reg": "TV"})),
    ("Sparse-view CT: FBP streaks", dict(problem="ct-sparse-view", method="fbp")),
    ("Sparse-view CT: TV + nonnegativity", dict(problem="ct-sparse-view", method="pdhg", params={"reg": "TV"})),
    ("Limited-angle CT: semiconvergence of SIRT", dict(problem="ct-limited-angle", method="sirt", params={"iters": 400})),
    ("MRI radial: TV compressed sensing", dict(problem="mri-radial", method="pdhg", params={"reg": "TV"})),
    ("MRI Cartesian: wavelet FISTA", dict(problem="mri-cartesian", method="fista", params={"basis": "db4"})),
    ("Compressed sensing: l1 recovers the support", dict(problem="compressed-sensing", method="fista")),
    ("Compressed sensing: OMP", dict(problem="compressed-sensing", method="omp")),
    ("Spike deconvolution: IRLS p = 0.5", dict(problem="spike-deconvolution", method="irls")),
    ("Gravity: Tikhonov + L-curve", dict(problem="gravity", method="tikhonov")),
    ("Shaw: TSVD and the Picard condition", dict(problem="shaw", method="tsvd")),
    ("Backward heat: Gaussian posterior UQ", dict(problem="backward-heat", method="gaussian-posterior",
                                                  params={"penalty": "gradient"})),
    ("Numerical differentiation: TV", dict(problem="differentiation", method="admm-tv")),
    ("Phase retrieval: amplitude flow", dict(problem="phase-retrieval", method="amplitude-flow")),
    ("Coefficient identification: Gauss-Newton + Laplace UQ", dict(problem="coefficient-id", method="gauss-newton")),
]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, theme="light"):
        super().__init__()
        self.setWindowTitle("Inverse Problem Toolbox")
        self.resize(1560, 940)
        self.settings = QtCore.QSettings("invtoolbox", "gui2")
        self.theme_name = theme
        self.problem: P.Problem | None = None
        self.data: P.Data | None = None
        self.result: M.Result | None = None
        self.worker: Worker | None = None
        self.sweep_worker: Worker | None = None
        self._pending = None
        self._lam_touched = False
        self._color_i = 0
        self._live_payload = None
        self._t_start = 0.0
        self._pin_next = False
        self._busy_problem = False
        self._running = False
        self._gen = 0                       # bumped whenever problem/data change; stale results are dropped
        self.user_truth = None              # (array, file name): your ground truth, kept across problems
        self.user_meas = None               # dict(key, size, b, sigma, name, path): your measurements
        self.custom = None                  # (problem, data, paths) from your own matrix
        self.debounce = QtCore.QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(140)
        self.debounce.timeout.connect(self.solve)
        self.paint_timer = QtCore.QTimer(self)
        self.paint_timer.setInterval(50)
        self.paint_timer.timeout.connect(self._paint_live)
        self._build()
        self.apply_theme(theme)
        self.load_problem(self.cb_problem.current_key() or "deblur-gaussian")

    # ================================================================== layout
    def _build(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.cb_example = QtWidgets.QComboBox()
        self.cb_example.addItem("Examples…")
        for name, _ in EXAMPLES:
            self.cb_example.addItem(name)
        self.cb_example.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cb_example.setMinimumWidth(max(330, self.cb_example.sizeHint().width()))
        self.cb_example.activated.connect(self.load_example)
        tb.addWidget(self.cb_example)
        tb.addSeparator()
        self.act_solve = tb.addAction("▶  Solve   F5", self.solve_and_pin)
        self.act_solve.setShortcut("F5")
        self.act_stop = tb.addAction("■  Stop   Esc", self.stop)
        self.act_stop.setShortcut("Esc")
        self.act_pin = tb.addAction("Pin to compare", self.pin_current)
        self.act_pin.setShortcut("Ctrl+P")
        tb.addSeparator()
        self.act_sweep = tb.addAction("λ sweep", lambda: self.run_sweep(self.lam_view.sp_n.value(), self.lam_view.sp_dec.value()))
        self.act_sweep.setShortcut("Ctrl+L")
        self.act_auto = tb.addAction("Auto λ (discrepancy)", lambda: self.run_sweep(15, 3.0, apply="discrepancy"))
        tb.addSeparator()
        tb.addAction("Save figure", self.save_figure).setShortcut("Ctrl+S")
        tb.addAction("Export result", self.export_result).setShortcut("Ctrl+E")
        tb.addAction("Save setup", self.save_setup)
        tb.addAction("Load setup", self.load_setup)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.act_theme = tb.addAction("Dark theme", self.toggle_theme)
        self.act_theme.setShortcut("Ctrl+T")

        split = QtWidgets.QSplitter(Qt.Horizontal)
        split.setHandleWidth(6)
        self.setCentralWidget(split)

        # --------------------------------------------------------- left column
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        scroll.setWidget(panel)
        left = QtWidgets.QVBoxLayout(panel)
        left.setContentsMargins(10, 10, 6, 10)
        left.setSpacing(10)

        f, lay = card("Problem")
        self.cb_problem = GroupedCombo()
        cats = P.categories()
        self.cb_problem.set_groups(cats, {k: s.title for k, s in P.REGISTRY.items()})
        self.cb_problem.set_key("deblur-gaussian")
        self.cb_problem.activated.connect(lambda _: self.load_problem(self.cb_problem.current_key()))
        lay.addWidget(self.cb_problem)
        self.eq_problem = EquationView()
        lay.addWidget(self.eq_problem)
        self.lbl_note = muted()
        lay.addWidget(self.lbl_note)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.cb_truth = QtWidgets.QComboBox()
        self.cb_truth.activated.connect(lambda _: self.load_problem(self.problem.key, keep_method=True))
        self.cb_size = QtWidgets.QComboBox()
        self.cb_size.activated.connect(lambda _: self.load_problem(self.problem.key, keep_method=True))
        self.cb_noise = QtWidgets.QComboBox()
        self.cb_noise.currentTextChanged.connect(self._on_noise_kind)
        from .widgets import LogSlider
        self.sl_noise = LogSlider(0.01, 1e-4, 0.5, log=True)
        self.sl_noise.valueChanged.connect(lambda _: self.resimulate())
        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setKeyboardTracking(False)
        self.sp_seed.valueChanged.connect(lambda _: self.resimulate())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.cb_truth, 2)
        row.addWidget(self.cb_size, 1)
        form.addRow("Truth / size", row)
        form.addRow("Noise model", self.cb_noise)
        self.lbl_noise = QtWidgets.QLabel("Noise level")
        form.addRow(self.lbl_noise, self.sl_noise)
        form.addRow("Seed", self.sp_seed)
        self.btn_truth = QtWidgets.QPushButton("Truth…")
        self.btn_truth.setToolTip("Use your own image or signal as the ground truth of this problem")
        self.btn_truth.clicked.connect(self._dlg_truth)
        self.btn_meas = QtWidgets.QPushButton("Measured b…")
        self.btn_meas.setToolTip("Reconstruct from your own measurements with this forward model (no ground truth)")
        self.btn_meas.clicked.connect(self._dlg_measured)
        self.btn_custom = QtWidgets.QPushButton("Matrix A, b…")
        self.btn_custom.setToolTip("Your own forward matrix and data vector")
        self.btn_custom.clicked.connect(self._dlg_custom)
        self.btn_clear_user = QtWidgets.QPushButton("Clear")
        self.btn_clear_user.setToolTip("Back to the built-in truth and simulated data")
        self.btn_clear_user.clicked.connect(self.clear_user_data)
        for b in (self.btn_truth, self.btn_meas, self.btn_custom, self.btn_clear_user):
            b.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)   # never widen the panel
        ugrid = QtWidgets.QGridLayout()
        ugrid.setHorizontalSpacing(4)
        ugrid.setVerticalSpacing(4)
        ugrid.addWidget(self.btn_truth, 0, 0)
        ugrid.addWidget(self.btn_meas, 0, 1)
        ugrid.addWidget(self.btn_custom, 1, 0)
        ugrid.addWidget(self.btn_clear_user, 1, 1)
        form.addRow("Your data", ugrid)
        lay.addLayout(form)
        self.lbl_user = muted("")
        self.lbl_user.hide()
        lay.addWidget(self.lbl_user)
        self.lbl_pinfo = QtWidgets.QLabel()
        self.lbl_pinfo.setObjectName("metric")
        self.lbl_pinfo.setWordWrap(True)
        lay.addWidget(self.lbl_pinfo)
        left.addWidget(f)

        f, lay = card("Method")
        self.cb_method = GroupedCombo()
        self.cb_method.activated.connect(lambda _: self.select_method(self.cb_method.current_key()))
        lay.addWidget(self.cb_method)
        self.eq_method = EquationView()
        lay.addWidget(self.eq_method)
        self.lbl_blurb = muted()
        lay.addWidget(self.lbl_blurb)
        self.form = ParamForm()
        self.form.changed.connect(self._on_param)
        lay.addWidget(self.form)
        row = QtWidgets.QHBoxLayout()
        self.chk_live = QtWidgets.QCheckBox("Live: re-solve on every change")
        self.chk_live.setChecked(True)
        row.addWidget(self.chk_live, 1)
        self.btn_solve = QtWidgets.QPushButton("Solve")
        self.btn_solve.setObjectName("primary")
        self.btn_solve.clicked.connect(self.solve_and_pin)
        row.addWidget(self.btn_solve)
        lay.addLayout(row)
        left.addWidget(f)

        self.lbl_metrics = QtWidgets.QLabel("–")
        self.lbl_metrics.setObjectName("metric")
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.hide()
        left.addStretch(1)
        split.addWidget(scroll)

        # --------------------------------------------------------- right tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(False)
        self.recon = ReconView()
        self.conv = ConvergenceView()
        self.lam_view = LambdaView()
        self.spec = SpectrumView()
        self.uq = UncertaintyView()
        self.cmp = CompareView()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.recon, "Reconstruction")
        self.tabs.addTab(self.conv, "Convergence")
        self.tabs.addTab(self.lam_view, "λ study")
        self.tabs.addTab(self.spec, "Spectrum")
        self.tabs.addTab(self.uq, "Uncertainty")
        self.tabs.addTab(self.cmp, "Compare")
        self.tabs.addTab(self.log, "Log")
        self.tabs.currentChanged.connect(self._on_tab)
        self.lam_view.apply_lam.connect(self._apply_lam)
        self.lam_view.apply_iters.connect(self._apply_iters)
        self.lam_view.run_sweep.connect(self.run_sweep)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 1130])

        sb = self.statusBar()
        self.lbl_status = QtWidgets.QLabel("Ready")
        sb.addWidget(self.lbl_status, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        self.progress.hide()
        sb.addPermanentWidget(self.progress)
        self.act_stop.setEnabled(False)

    # ================================================================== theme
    def apply_theme(self, name):
        self.theme_name = name
        app = QtWidgets.QApplication.instance()
        self.t = apply_theme(app, name)
        self.series = SERIES_DARK if name == "dark" else SERIES
        for v in (self.recon, self.conv, self.lam_view, self.spec, self.uq, self.cmp):
            v.set_theme(self.t)
        self.act_theme.setText("Light theme" if name == "dark" else "Dark theme")
        for w in (self.eq_problem, self.eq_method):
            w.update()

    def toggle_theme(self):
        self.apply_theme("light" if self.theme_name == "dark" else "dark")
        if self.problem is not None:
            self.conv.setup()
            self._setup_views()
            self._show_result(self.result, final=True)
            self.cmp.redraw()

    # ================================================================== problem
    def _spec(self, key):
        if key == "custom":
            n = self.custom[0].n
            return SimpleNamespace(truth="user", truths=("user",), sizes=(n,), size=n, title=self.custom[0].title)
        return P.REGISTRY[key]

    def _make_problem(self, key):
        """Built-in problem, with your truth and / or measurements applied when loaded."""
        if key == "custom":
            return self.custom[0]
        size = int(self.cb_size.currentText())
        p = P.build(key, size, self.cb_truth.currentText())
        if self.user_truth is not None:
            p = U.with_truth(p, self.user_truth[0], name=self.user_truth[1])
        um = self.user_meas
        if um is not None and (um["key"], um["size"]) != (key, size):
            self.log_msg(f"measured data {um['name']} belong to {um['key']} (size {um['size']}): cleared")
            self.user_meas = um = None
        if um is not None:
            p, _ = U.with_measured(p, um["b"], um["sigma"], name=um["name"],
                                   truth=p.x_true.reshape(p.shape) if self.user_truth is not None else None)
        return p

    def _refresh_problem_list(self):
        cats = dict(P.categories())
        titles = {k: s.title for k, s in P.REGISTRY.items()}
        if self.custom is not None:
            cats["User data"] = ["custom"]
            titles["custom"] = self.custom[0].title
        self.cb_problem.set_groups(cats, titles)

    def load_problem(self, key, keep_method=False, truth=None, size=None, noise=None):
        spec = self._spec(key)
        same = self.problem is not None and self.problem.key == key
        if not same or truth is not None or size is not None:
            self.cb_truth.blockSignals(True)
            self.cb_size.blockSignals(True)
            cur_truth = truth or (self.cb_truth.currentText() if same else spec.truth)
            cur_size = size or (int(self.cb_size.currentText()) if same and self.cb_size.currentText() else spec.size)
            self.cb_truth.clear()
            self.cb_truth.addItems(list(spec.truths))
            self.cb_truth.setCurrentText(cur_truth if cur_truth in spec.truths else spec.truth)
            self.cb_size.clear()
            self.cb_size.addItems([str(s) for s in spec.sizes])
            self.cb_size.setCurrentText(str(cur_size if cur_size in spec.sizes else spec.size))
            self.cb_truth.blockSignals(False)
            self.cb_size.blockSignals(False)
        self.stop()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            p = self._make_problem(key)
        except Exception as exc:  # noqa: BLE001  (bad user file: report, keep the old problem)
            self._user_error(exc)
            if self.user_meas is not None:
                self.user_meas = None
            if self.user_truth is not None:
                self.user_truth = None
            self._update_user_label()
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        changed_key = self.problem is None or self.problem.key != key
        self._gen += 1
        self._live_payload = None
        self.result = None
        self.problem = p
        self.cb_problem.set_key(key)
        self.eq_problem.set_equations(p.equations)
        self.lbl_note.setText(p.note)
        self._busy_problem = True
        self.cb_noise.blockSignals(True)
        cur_kind = self.cb_noise.currentText()
        self.cb_noise.clear()
        self.cb_noise.addItems(list(p.noise_kinds))
        if noise is not None:
            self.cb_noise.setCurrentText(noise[0])
            self.sl_noise.setValue(noise[1])
        elif changed_key or cur_kind not in p.noise_kinds:
            self.cb_noise.setCurrentText(p.noise_default[0])
            self.sl_noise.setValue(p.noise_default[1])
        else:
            self.cb_noise.setCurrentText(cur_kind)
        self.cb_noise.blockSignals(False)
        self._noise_label()
        measured = "measured" in p.meta
        for w in (self.cb_noise, self.sl_noise, self.sp_seed):
            w.setEnabled(not measured)
        if measured:
            self.sl_noise.blockSignals(True)
            self.sl_noise.setValue(p.meta["measured"].level)
            self.sl_noise.blockSignals(False)
        for w in (self.cb_truth, self.cb_size):
            w.setEnabled(key != "custom")
        self._busy_problem = False
        # methods
        why = M.applicable(p)
        self.cb_method.set_groups(M.groups(), {k: m.label for k, m in M.METHODS.items()}, why)
        self.cmp.set_problem(p)
        self.data = p.simulate(self.cb_noise.currentText(), self.sl_noise.value(), self.sp_seed.value())
        self._setup_views()
        mk = self.cb_method.current_key() if keep_method else None
        if not mk or why.get(mk):
            mk, extra = RECOMMENDED.get(key, (self.cb_method.first_enabled(), {}))
            self.select_method(mk, extra, solve=False)
        else:
            self.select_method(mk, self.form.values(), solve=False, keep_lam=self._lam_touched)
        self._update_pinfo()
        self._update_user_label()
        self.log_msg(f"problem {key}: n = {p.n}, m = {p.m}, {self.data.kind} noise {fmt(self.data.level)}")
        self.request_solve(now=True)

    def _setup_views(self):
        self.recon.setup(self.problem, self.data)
        self.recon.set_chips([])
        self.conv.setup()
        self.lam_view.blank("Press “Run λ sweep” to scan the regularization parameter.")
        for v in (self.conv, self.lam_view, self.spec, self.uq):
            v.dirty = True
        self._on_tab(self.tabs.currentIndex())

    def _noise_label(self):
        k = self.cb_noise.currentText()
        self.lbl_noise.setText({"gaussian": "Noise ‖e‖/‖Ax‖", "poisson": "1/√(peak photons)",
                                "salt-pepper": "Corrupted fraction", "measured": "Noise ‖e‖/‖b‖"}.get(k, "Noise level"))

    def _on_noise_kind(self, kind):
        if self._busy_problem or not kind or kind == "measured":
            return
        if kind == "poisson":
            self.sl_noise.setValue(0.1)
        elif kind == "salt-pepper":
            self.sl_noise.setValue(0.1)
        else:
            self.sl_noise.setValue(self.problem.noise_default[1] if self.problem.noise_default[0] == "gaussian" else 0.02)
        self._noise_label()
        if kind == "salt-pepper" and self.cb_method.current_key() == "pdhg":
            self.form.set_value("data_term", "L1")
        if kind == "poisson" and self.cb_method.current_key() == "pdhg" and self.problem.A is not None and self.problem.A.nonneg:
            self.form.set_value("data_term", "KL")
        self.resimulate()

    def resimulate(self):
        if self.problem is None or self._busy_problem:
            return
        self._gen += 1
        self._live_payload = None
        self.data = self.problem.simulate(self.cb_noise.currentText(), self.sl_noise.value(), self.sp_seed.value())
        self.recon.set_data(self.data)
        self.lam_view.blank("Data changed: run the λ sweep again.")
        self.spec.dirty = True
        if not self._lam_touched:
            self._set_default_lam()
        self._update_pinfo()
        self.request_solve()

    def _update_pinfo(self):
        p, d = self.problem, self.data
        txt = f"n = {p.n}   m = {p.m}   δ = ‖e‖ ≈ {d.delta:.3g}"
        if p.meta.get("sigma_estimated"):
            txt += " (estimated)"
        if p.linear and p.A.has_svd and p.A._svd is not None:
            s = p.A._svd[1]
            txt += f"\ncond(A) = {s[0] / max(s[-1], 1e-300):.1e}"
        if d.kind == "poisson":
            txt += f"   peak ≈ {1 / d.level ** 2:.0f} photons"
        self.lbl_pinfo.setText(txt)

    # ================================================================== method
    def select_method(self, key, extra=None, solve=True, keep_lam=False):
        if key is None:
            return
        m = M.METHODS[key]
        self.cb_method.set_key(key)
        self.eq_method.set_equations(m.equations)
        self.lbl_blurb.setText(m.blurb)
        vals = m.defaults()
        old = self.form.values() if keep_lam else {}
        vals.update({k: v for k, v in (extra or {}).items() if k in vals})
        if keep_lam and "lam" in old and "lam" in vals:
            vals["lam"] = old["lam"]
        elif "lam" in vals:
            vals["lam"] = default_lam(key, self.problem, self.data, vals)
            self._lam_touched = False
        self.form.build(m.params, vals)
        has_lam = "lam" in vals and key != "richardson-lucy"
        self.act_sweep.setEnabled(has_lam)
        self.act_auto.setEnabled(has_lam)
        self.lam_view.set_mode(has_lam, m.iterative)
        if not has_lam and m.iterative:
            self.lam_view.blank("Run the method: its iterations are the regularization parameter.")
        if solve:
            self.request_solve(now=True)

    def _set_default_lam(self):
        key = self.cb_method.current_key()
        vals = self.form.values()
        if "lam" in vals and key != "richardson-lucy":
            self.form.set_value("lam", default_lam(key, self.problem, self.data, vals))

    def _on_param(self, k):
        if k == "lam":
            self._lam_touched = True
        elif k in ("penalty", "basis", "data_term", "reg", "operator") and not self._lam_touched:
            self._set_default_lam()
        self.request_solve()

    def _apply_lam(self, lam):
        self.form.set_value("lam", lam)
        self._lam_touched = True
        self.log_msg(f"λ ← {fmt(lam)}")
        self.request_solve(now=True)

    def _apply_iters(self, k):
        key = "iters" if "iters" in self.form.widgets else None
        if key:
            self.form.set_value(key, k)
            self.log_msg(f"iterations ← {k}")
            self.request_solve(now=True)

    # ================================================================== solving
    def request_solve(self, now=False):
        if not self.chk_live.isChecked() and not now:
            return
        if now:
            self.debounce.stop()
            self.solve()
        else:
            self.debounce.start()

    def solve_and_pin(self):
        self._pin_next = True
        self.solve()

    def solve(self):
        if self.problem is None:
            return
        key = self.cb_method.current_key()
        if key is None or M.applicable(self.problem).get(key):
            return
        job = (key, self.form.values())
        if self.worker is not None and self.worker.isRunning():
            self._pending = job
            self.worker.stop()
            return
        self._start(job)

    def _start(self, job):
        if self.worker is not None:
            self.worker.wait()          # the previous thread has emitted finished; let it exit before release
        key, params = job
        problem, data = self.problem, self.data
        self._job = job
        gen = self._gen

        def fn(emit, stopped):
            def cb(k, x, hist):
                emit((gen, k, x.copy(), {kk: list(v) for kk, v in hist.items()}))
            res = M.run(key, problem, data, params, callback=cb, stop=stopped, every=0.08)
            res.info["_gen"] = gen
            return res

        w = Worker(fn, tag=key)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(lambda w=w: self._on_cancelled() if w is self.worker else None)
        w.finished.connect(lambda w=w: self._after_worker(w))
        self.worker = w
        self._running = True
        self._t_start = time.perf_counter()
        self._busy(True)
        self.lbl_status.setText(f"Solving with {M.METHODS[key].label}…")
        self.paint_timer.start()
        w.start()

    def stop(self):
        self._pending = None
        self.debounce.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        if self.sweep_worker is not None and self.sweep_worker.isRunning():
            self.sweep_worker.stop()

    def _busy(self, on):
        self.act_stop.setEnabled(on)
        if on:
            self.progress.setRange(0, 0)
            self.progress.show()
        else:
            self.progress.hide()

    def _on_progress(self, payload):
        self._live_payload = payload

    def _paint_live(self):
        if self._live_payload is None:
            return
        gen, k, x, hist = self._live_payload
        self._live_payload = None
        if gen != self._gen:
            return
        self.recon.update_x(x)
        if self.tabs.currentWidget() is self.conv:
            self.conv.update_hist(hist, self.data.delta)
        e = hist["err"][-1] if hist["err"] else float("nan")
        err = f" · rel. error {e:.4f}" if np.isfinite(e) else ""
        self.lbl_status.setText(f"{M.METHODS[self._job[0]].label} · iteration {k + 1}{err} · "
                                f"{time.perf_counter() - self._t_start:.1f} s")

    def _on_done(self, res: M.Result):
        self._live_payload = None
        if res.info.pop("_gen", self._gen) != self._gen:
            return
        self.result = res
        self._show_result(res, final=True)
        i = res.info
        tail = f"rel. error {i['rel_err']:.4f}" if np.isfinite(i["rel_err"]) else f"‖r‖/δ {i['discrepancy']:.2f}"
        self.lbl_status.setText(f"{M.METHODS[res.method].label} done in {res.seconds:.2f} s · {tail}")
        if self._pin_next:
            self._pin_next = False
            self.pin_current()

    def _show_result(self, res, final=False):
        if res is None:
            return
        self.recon.update_x(res.x, res.std if not self.problem.is_image else None)
        i = res.info
        lines = [f"relative error  {i['rel_err']:.4f}", f"PSNR            {i['psnr']:.2f} dB"]
        if self.problem.is_image:
            lines.append(f"SSIM            {i['ssim']:.4f}")
        lines += [f"‖F(x̂) − b‖/δ    {i['discrepancy']:.3f}",
                  f"iterations      {len(res.hist.get('err', []))}", f"time            {res.seconds:.3f} s"]
        extra = {k: v for k, v in i.items() if k in ("k", "path", "support", "uq", "stopped_at", "samples")}
        for k, v in extra.items():
            lines.append(f"{k:<15} {v}")
        self.lbl_metrics.setText("\n".join(lines))
        if self.problem.has_truth:
            chips = [f"error {i['rel_err']:.4f}", f"PSNR {i['psnr']:.2f} dB"]
            if self.problem.is_image:
                chips.append(f"SSIM {i['ssim']:.3f}")
        else:
            chips = ["no ground truth"]
        chips += [f"‖r‖/δ {i['discrepancy']:.2f}", f"{len(res.hist.get('err', []))} it · {res.seconds:.2f} s"]
        self.recon.set_chips(chips)
        self.lbl_metrics.setToolTip("\n".join(lines))
        for v in (self.conv, self.lam_view, self.spec, self.uq):
            v.dirty = True
        self._on_tab(self.tabs.currentIndex())

    def _on_failed(self, msg):
        self.lbl_status.setText("Solver failed: " + msg.splitlines()[0])
        self.log_msg(msg)

    def _on_cancelled(self):
        self.lbl_status.setText("Cancelled")

    def busy(self):
        """True while a solve or sweep is queued, running, or has results not yet delivered."""
        return (self._running or self._pending is not None or self.debounce.isActive()
                or (self.sweep_worker is not None and self.sweep_worker.isRunning()))

    def _after_worker(self, w=None):
        if w is not None and w is not self.worker:      # a superseded worker finishing late: not ours
            return
        self._running = False
        self.paint_timer.stop()
        self._paint_live()
        self._busy(False)
        if self._pending is not None:
            job = self._pending
            self._pending = None
            if job[0] == self.cb_method.current_key():
                job = (job[0], self.form.values())
            self._start(job)

    # ================================================================== tabs
    def _on_tab(self, idx):
        w = self.tabs.widget(idx)
        if not getattr(w, "dirty", False) or self.problem is None:
            return
        w.dirty = False
        res = self.result if (self.result is not None and self.result.x.size == self.problem.n) else None
        if w is self.conv:
            others = [(f"#{r['id']} {r['label']}", r["res"].hist, r["color"]) for r in self.cmp.selected()] if self.cmp.runs else []
            self.conv.update_hist(res.hist if res else {}, self.data.delta, others)
        elif w is self.spec:
            params = dict(res.params) if res else {}
            if res is not None and "k" in res.info:
                params["k_used"] = res.info["k"]
            self.spec.show(self.problem, self.data, res.method if res else None, params)
            self._update_pinfo()
        elif w is self.uq:
            self.uq.show(self.problem, res)
        elif w is self.lam_view:
            key = self.cb_method.current_key()
            if key is None:
                w.dirty = True
                return
            m = M.METHODS[key]
            if "lam" not in m.defaults() and m.iterative and res is not None and res.method == key:
                self.lam_view.show_iterations(res.hist, self.data.delta)

    # ================================================================== lambda sweep
    def run_sweep(self, n=15, decades=3.0, apply=None):
        key = self.cb_method.current_key()
        params = self.form.values()
        if "lam" not in params:
            return
        if self.sweep_worker is not None:
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        center = params["lam"]
        lams = PC.lam_grid(center, decades, n)
        problem, data = self.problem, self.data
        if key == "richardson-lucy":
            return

        def fn(emit, stopped):
            return PC.sweep(key, problem, data, {k: v for k, v in params.items() if k != "lam"}, lams, stop=stopped,
                            progress=lambda i, nn, _: emit((i, nn)))

        w = Worker(fn, tag="sweep")
        w.progress.connect(lambda pr: (self.progress.setRange(0, pr[1]), self.progress.setValue(pr[0] + 1),
                                       self.lbl_status.setText(f"λ sweep {pr[0] + 1}/{pr[1]}…")))
        w.finished_ok.connect(lambda sw: self._sweep_done(sw, apply))
        w.failed.connect(self._on_failed)
        w.finished.connect(lambda: self.progress.hide())
        self.sweep_worker = w
        self.progress.setRange(0, n)
        self.progress.show()
        self.tabs.setCurrentWidget(self.lam_view)
        w.start()

    def _sweep_done(self, sw, apply):
        self.lam_view.show_sweep(sw, self.data.delta, self.form.values().get("lam"))
        ch = ", ".join(f"{r} λ = {fmt(sw['lam'][i])}" for r, i in sw["choices"].items())
        self.log_msg(f"λ sweep ({self.cb_method.current_key()}): {ch}")
        self.lbl_status.setText("λ sweep done")
        if apply:
            self._apply_lam(float(sw["lam"][sw["choices"][apply]]))

    # ================================================================== compare / export
    def pin_current(self):
        res = self.result
        if res is None:
            return
        col = self.series[self._color_i % len(self.series)]
        self._color_i += 1
        m = M.METHODS[res.method]
        sets = []
        for p in m.params:
            v = res.params.get(p.key)
            if v is None:
                continue
            sets.append(f"{p.key}={fmt(v) if isinstance(v, float) else v}")
        noise = f"{self.data.kind} {fmt(self.data.level)}"
        self.cmp.add(m.label, noise + " · " + ", ".join(sets), res, col)
        self.conv.dirty = True

    def save_figure(self):
        w = self.tabs.currentWidget()
        canvas = getattr(w, "canvas", None)
        if canvas is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save figure", f"{self.problem.key}.png",
                                                        "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            canvas.fig.savefig(path, dpi=200)
            self.log_msg(f"saved {path}")

    def export_result(self):
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export result", f"{self.problem.key}_{self.result.method}.npz",
                                                        "NumPy archive (*.npz)")
        if not path:
            return
        r = self.result
        np.savez(path, x=r.x.reshape(self.problem.shape), x_true=self.problem.x_true.reshape(self.problem.shape),
                 b=self.data.b, std=(r.std if r.std is not None else np.array([])),
                 **{f"hist_{k}": np.asarray(v) for k, v in r.hist.items()})
        meta = {"problem": self.problem.key, "method": r.method, "params": r.params,
                "noise": [self.data.kind, self.data.level, self.data.seed],
                "metrics": {k: v for k, v in r.info.items() if isinstance(v, (int, float, str))}}
        with open(path.rsplit(".", 1)[0] + ".json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        self.log_msg(f"exported {path}")

    def setup_dict(self):
        d = {"problem": self.problem.key, "truth": self.cb_truth.currentText(), "size": int(self.cb_size.currentText()),
             "noise": [self.cb_noise.currentText(), self.sl_noise.value()], "seed": self.sp_seed.value(),
             "method": self.cb_method.current_key(), "params": self.form.values()}
        user = {}
        if self.user_truth is not None and len(self.user_truth) > 2:
            user["truth_file"] = self.user_truth[2]
        if self.user_meas is not None:
            user["data_file"], user["sigma"] = self.user_meas["path"], self.user_meas["sigma"]
        if self.custom is not None and self.problem.key == "custom":
            user["custom"] = self.custom[2]
        if user:
            d["user"] = user
        return d

    def apply_setup(self, s):
        u = s.get("user") or {}
        try:
            if u.get("custom"):
                c = u["custom"]
                self.load_custom(c["A"], c["b"], c.get("truth"), c.get("shape"), c.get("sigma"), activate=False)
            if u.get("truth_file"):
                self.load_user_truth(u["truth_file"], activate=False)
        except Exception as exc:  # noqa: BLE001
            self._user_error(exc)
        self.sp_seed.blockSignals(True)
        self.sp_seed.setValue(int(s.get("seed", 0)))
        self.sp_seed.blockSignals(False)
        self.chk_live.blockSignals(True)
        self.load_problem(s["problem"], truth=s.get("truth"), size=s.get("size"), noise=tuple(s["noise"]) if s.get("noise") else None)
        if u.get("data_file") and s["problem"] != "custom":
            try:
                self.load_user_measured(u["data_file"], u.get("sigma"))
            except Exception as exc:  # noqa: BLE001
                self._user_error(exc)
        self.select_method(s["method"], s.get("params", {}), solve=False, keep_lam=False)
        if "lam" in s.get("params", {}):
            self.form.set_value("lam", s["params"]["lam"])
            self._lam_touched = True
        self.chk_live.blockSignals(False)
        self.request_solve(now=True)

    def save_setup(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save setup", "setup.json", "JSON (*.json)")
        if path:
            with open(path, "w") as fh:
                json.dump(self.setup_dict(), fh, indent=2)

    def load_setup(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load setup", "", "JSON (*.json)")
        if path:
            with open(path) as fh:
                self.apply_setup(json.load(fh))

    def load_example(self, idx):
        if idx <= 0:
            return
        name, ex = EXAMPLES[idx - 1]
        P_ = P.REGISTRY[ex["problem"]]
        noise = ex.get("noise")
        self.load_problem(ex["problem"], truth=P_.truth, size=P_.size, noise=noise)
        self.select_method(ex["method"], ex.get("params", {}), solve=True)
        self.log_msg(f"example: {name}")
        self.cb_example.setCurrentIndex(0)

    # ================================================================== your data
    def load_user_truth(self, path, activate=True):
        """Your image / signal becomes the ground truth (resampled to the grid) of every built-in problem."""
        arr = U.load_array(path)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        self.user_truth = (np.asarray(arr, float), os.path.basename(path), path)
        self.log_msg(f"truth ← {path} {np.shape(arr)}")
        if activate:
            key = self.problem.key if self.problem is not None and self.problem.key != "custom" else "deblur-gaussian"
            self.load_problem(key, keep_method=True)

    def load_user_measured(self, path, sigma=None):
        """Your measurements for the current built-in forward model; the truth becomes unknown."""
        if self.problem is None or self.problem.key == "custom":
            raise ValueError("pick a built-in problem first: its forward model is applied to your data")
        b = U.load_array(path)
        key, size = self.problem.key, int(self.cb_size.currentText())
        P_ = P.build(key, size, self.cb_truth.currentText())
        U._as_data_vector(P_, b)                       # validate the shape now, with a clear message
        self.user_meas = dict(key=key, size=size, b=b, sigma=sigma, name=os.path.basename(path), path=path)
        self.log_msg(f"measured data ← {path} {np.shape(b)}" + ("" if sigma is None else f", σ = {sigma:g}"))
        self.load_problem(key, keep_method=True)

    def load_custom(self, A_path, b_path, truth_path=None, shape=None, sigma=None, activate=True):
        """Your own matrix A and data b (and optionally a truth for validation)."""
        A = U.load_array(A_path)
        b = U.load_array(b_path)
        xt = U.load_array(truth_path).ravel() if truth_path else None
        shp = tuple(int(v) for v in str(shape).replace("x", ",").split(",") if v.strip()) if shape else None
        p, d = U.custom_problem(A, b, xt, shp, sigma, title=f"Your problem: {os.path.basename(A_path)}")
        self.custom = (p, d, dict(A=A_path, b=b_path, truth=truth_path, shape=shape, sigma=sigma))
        self._refresh_problem_list()
        self.log_msg(f"custom problem ← A {A_path} {np.shape(A)}, b {b_path}")
        if activate:
            self.load_problem("custom")

    def clear_user_data(self):
        was_custom = self.problem is not None and self.problem.key == "custom"
        self.user_truth = self.user_meas = None
        self.custom = None
        self._refresh_problem_list()
        self.log_msg("your data cleared")
        self.load_problem("deblur-gaussian" if was_custom else self.problem.key, keep_method=not was_custom)

    def _update_user_label(self):
        p = self.problem
        parts = []
        if p is not None and p.key == "custom":
            parts.append(f"Your matrix {os.path.basename(self.custom[2]['A'])} and data {os.path.basename(self.custom[2]['b'])}")
        if self.user_truth is not None and (p is None or p.key != "custom"):
            parts.append(f"Truth: {self.user_truth[1]}")
        if self.user_meas is not None:
            parts.append(f"Measured data: {self.user_meas['name']}")
        if p is not None and p.meta.get("sigma_estimated"):
            parts.append("noise level estimated from the data")
        self.lbl_user.setText(" · ".join(parts))
        self.lbl_user.setVisible(bool(parts))
        self.btn_clear_user.setEnabled(bool(self.user_truth or self.user_meas or self.custom))

    def _user_error(self, exc):
        msg = f"Could not use that file: {exc}"
        self.log_msg(msg)
        self.lbl_status.setText(msg)
        if not os.environ.get("INVTOOLBOX_NO_EXEC"):
            QtWidgets.QMessageBox.warning(self, "Your data", msg)

    def _open(self, title):
        last = self.settings.value("data_dir", "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, last, DATA_FILES)
        if path:
            self.settings.setValue("data_dir", os.path.dirname(path))
        return path

    def _dlg_truth(self):
        path = self._open("Ground truth: image or signal")
        if path:
            try:
                self.load_user_truth(path)
            except Exception as exc:  # noqa: BLE001
                self._user_error(exc)

    def _ask_sigma(self):
        txt, ok = QtWidgets.QInputDialog.getText(
            self, "Noise level", "Noise std per data entry, in the units of your data.\n"
                                 "Leave blank to estimate it from the data.")
        if not ok:
            return False, None
        txt = txt.strip()
        return True, (float(txt) if txt else None)

    def _dlg_measured(self):
        path = self._open(f"Measured data for {self.problem.title}")
        if not path:
            return
        ok, sigma = self._ask_sigma()
        if not ok:
            return
        try:
            self.load_user_measured(path, sigma)
        except Exception as exc:  # noqa: BLE001
            self._user_error(exc)

    def _dlg_custom(self):
        A_path = self._open("Forward matrix A (m × n)")
        if not A_path:
            return
        b_path = self._open("Data vector b (m entries)")
        if not b_path:
            return
        shape, ok = QtWidgets.QInputDialog.getText(self, "Shape of x", "Display shape of x, e.g. 64,64 for an image.\n"
                                                                       "Leave blank for a vector.")
        if not ok:
            return
        ok, sigma = self._ask_sigma()
        if not ok:
            return
        truth = None
        if QtWidgets.QMessageBox.question(self, "Ground truth", "Do you have a ground truth x to compare against?") \
                == QtWidgets.QMessageBox.Yes:
            truth = self._open("Ground truth x (n entries)") or None
        try:
            self.load_custom(A_path, b_path, truth, shape.strip() or None, sigma)
        except Exception as exc:  # noqa: BLE001
            self._user_error(exc)

    def log_msg(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def closeEvent(self, ev):
        self.stop()
        for w in (self.worker, self.sweep_worker):
            if w is not None:
                w.wait()
        super().closeEvent(ev)


def _install_excepthook():
    """PyQt aborts on exceptions escaping a slot; log them to stderr and the window's log instead."""
    import traceback

    def hook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        sys.stderr.write(msg)
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, MainWindow):
                w.log_msg(msg)
                w.lbl_status.setText(f"Error: {value}")
    sys.excepthook = hook


def main(argv=None):
    import os
    argv = sys.argv if argv is None else argv
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    _install_excepthook()
    theme = "dark" if "--dark" in argv else "light"
    win = MainWindow(theme)
    win.show()
    if not hasattr(sys, "ps1") and not os.environ.get("INVTOOLBOX_NO_EXEC"):
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()
