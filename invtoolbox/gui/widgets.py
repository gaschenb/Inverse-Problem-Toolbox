"""Reusable Qt widgets: theme, vector-rendered equations, parameter forms, log sliders, workers."""
from __future__ import annotations

import re
import threading
import traceback

import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# ----------------------------------------------------------------------------- theme
THEMES = {
    "light": dict(bg="#f4f5f7", panel="#ffffff", card="#ffffff", border="#dcdfe4", text="#1f2328", muted="#667085",
                  accent="#2563eb", accent_hover="#1d4ed8", accent_text="#ffffff", input="#ffffff",
                  sel="#dbe7ff", plot_bg="#ffffff", plot_fg="#1f2328", grid="#e6e8eb", bad="#d0d4da"),
    "dark": dict(bg="#16181d", panel="#1d2027", card="#22252d", border="#343844", text="#e6e8ee", muted="#9aa3b2",
                 accent="#4f8cff", accent_hover="#6a9dff", accent_text="#0d1117", input="#2a2e37",
                 sel="#2c3c5e", plot_bg="#1d2027", plot_fg="#d7dbe3", grid="#2f3440", bad="#3a3f4b"),
}

SERIES = ["#2563eb", "#e8590c", "#0f9d58", "#c2255c", "#7048e8", "#0b7285", "#d9480f", "#5c940d", "#862e9c",
          "#1864ab"]
SERIES_DARK = ["#5b9bff", "#ff922b", "#40c057", "#f06595", "#9775fa", "#22b8cf", "#ffa94d", "#94d82d", "#da77f2",
               "#74c0fc"]


def qss(t):
    return f"""
QMainWindow, QWidget {{ background: {t['bg']}; color: {t['text']}; font-size: 10pt; }}
QToolBar {{ background: {t['panel']}; border: none; border-bottom: 1px solid {t['border']}; padding: 4px; spacing: 4px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; }}
QToolBar QToolButton:hover {{ background: {t['sel']}; }}
QToolBar QToolButton:disabled {{ color: {t['muted']}; }}
QFrame#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 10px; }}
QLabel#cardTitle {{ font-weight: 600; font-size: 10.5pt; background: transparent; }}
QLabel#muted {{ color: {t['muted']}; background: transparent; }}
QLabel#metric {{ font-family: Consolas, 'DejaVu Sans Mono', monospace; background: transparent; }}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {t['input']}; border: 1px solid {t['border']};
    border-radius: 6px; padding: 3px 6px; min-height: 22px; selection-background-color: {t['accent']}; }}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t['accent']}; }}
QComboBox QAbstractItemView {{ background: {t['panel']}; border: 1px solid {t['border']}; selection-background-color: {t['sel']};
    selection-color: {t['text']}; outline: 0; }}
QPushButton {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {t['accent']}; }}
QPushButton:disabled {{ color: {t['muted']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']}; border: none; font-weight: 600; padding: 7px 14px; }}
QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {t['border']}; color: {t['muted']}; }}
QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 8px; background: {t['panel']}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 7px 14px; border: none; color: {t['muted']}; }}
QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['accent']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QTableWidget {{ background: {t['panel']}; gridline-color: {t['border']}; border: none; selection-background-color: {t['sel']};
    selection-color: {t['text']}; }}
QHeaderView::section {{ background: {t['bg']}; border: none; border-bottom: 1px solid {t['border']}; padding: 5px; font-weight: 600; }}
QPlainTextEdit {{ background: {t['panel']}; border: none; font-family: Consolas, 'DejaVu Sans Mono', monospace; }}
QStatusBar {{ background: {t['panel']}; border-top: 1px solid {t['border']}; }}
QStatusBar QLabel {{ background: transparent; padding: 0 8px; }}
QProgressBar {{ border: 1px solid {t['border']}; border-radius: 4px; background: {t['input']}; max-height: 10px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
QSplitter::handle {{ background: {t['bg']}; }}
QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 4px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t['muted']}; border-radius: 4px; background: {t['input']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QLabel#chip {{ background: {t['sel']}; border-radius: 9px; padding: 2px 9px; font-family: Consolas, 'DejaVu Sans Mono', monospace; }}
QMenu {{ background: {t['panel']}; border: 1px solid {t['border']}; }}
QMenu::item:selected {{ background: {t['sel']}; }}
"""


def apply_theme(app, name):
    t = THEMES[name]
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    pal.setColor(QtGui.QPalette.Window, c(t["bg"]))
    pal.setColor(QtGui.QPalette.WindowText, c(t["text"]))
    pal.setColor(QtGui.QPalette.Base, c(t["input"]))
    pal.setColor(QtGui.QPalette.AlternateBase, c(t["bg"]))
    pal.setColor(QtGui.QPalette.Text, c(t["text"]))
    pal.setColor(QtGui.QPalette.Button, c(t["input"]))
    pal.setColor(QtGui.QPalette.ButtonText, c(t["text"]))
    pal.setColor(QtGui.QPalette.Highlight, c(t["accent"]))
    pal.setColor(QtGui.QPalette.HighlightedText, c(t["accent_text"]))
    pal.setColor(QtGui.QPalette.ToolTipBase, c(t["panel"]))
    pal.setColor(QtGui.QPalette.ToolTipText, c(t["text"]))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, c(t["muted"]))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, c(t["muted"]))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c(t["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(qss(t))
    matplotlib.rcParams.update({
        "figure.facecolor": t["plot_bg"], "axes.facecolor": t["plot_bg"], "savefig.facecolor": t["plot_bg"],
        "axes.edgecolor": t["border"], "axes.labelcolor": t["plot_fg"], "text.color": t["plot_fg"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "grid.color": t["grid"], "axes.grid": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "mathtext.fontset": "cm",
        "axes.prop_cycle": matplotlib.cycler(color=SERIES_DARK if name == "dark" else SERIES),
    })
    return t


# ----------------------------------------------------------------------------- equations
def _tex_fix(tex: str) -> str:
    return re.sub(r"\\(mathbf|mathrm|mathit|boldsymbol)\s+([A-Za-z])", r"\\\1{\2}", tex.strip().strip("$"))


_PATHS: dict = {}


def tex_path(tex: str):
    """Vector outline (QPainterPath, y down) of a mathtext string, cached; None if it does not parse."""
    if tex in _PATHS:
        return _PATHS[tex]
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    try:
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            tp = TextPath((0, 0), f"${_tex_fix(tex)}$", size=20, prop=FontProperties(family="DejaVu Sans"))
            verts, codes = tp.vertices, tp.codes
    except Exception:
        _PATHS[tex] = None
        return None
    path = QtGui.QPainterPath()
    i = 0
    pt = lambda k: QtCore.QPointF(verts[k, 0], -verts[k, 1])
    while i < len(codes):
        c = codes[i]
        if c == 1:
            path.moveTo(pt(i)); i += 1
        elif c == 2:
            path.lineTo(pt(i)); i += 1
        elif c == 3:
            path.quadTo(pt(i), pt(i + 1)); i += 2
        elif c == 4:
            path.cubicTo(pt(i), pt(i + 1), pt(i + 2)); i += 3
        else:
            path.closeSubpath(); i += 1
    path.setFillRule(Qt.WindingFill)
    _PATHS[tex] = path
    return path


class EquationView(QtWidgets.QWidget):
    """Equations drawn as filled glyph outlines from matplotlib's mathtext engine: crisp at any DPI,
    no LaTeX install, coloured from the palette, scaled down (never up) to fit the width."""

    def __init__(self, parent=None, scale=1.0):
        super().__init__(parent)
        self._lines = []
        self._scale = scale
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_equations(self, lines):
        self._lines = [(tex_path(t), t) for t in (lines or []) if t and t.strip()]
        self.setFixedHeight(self.heightForWidth(max(self.width(), 300)))
        self.update()

    def _layout(self, width):
        fm = QtGui.QFontMetricsF(self.font())
        W = max(width - 8, 40)
        k0 = fm.height() * 1.25 * self._scale / 20.0
        rows, y = [], 4.0
        for p, t in self._lines:
            if p is None or p.isEmpty():
                h = fm.height()
                rows.append((p, t, None, y, h, 1.0))
            else:
                br = p.boundingRect()
                k = min(k0, W / max(br.width(), 1.0))
                h = br.height() * k
                rows.append((p, t, br, y, h, k))
            y += h + fm.height() * 0.5
        return rows, y, W

    def heightForWidth(self, width):
        _, y, _ = self._layout(width)
        return int(y + 2) if self._lines else 0

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        h = self.heightForWidth(self.width())
        if h != self.height():
            self.setFixedHeight(h)

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        ink = self.palette().color(QtGui.QPalette.WindowText)
        rows, _, W = self._layout(self.width())
        for path, tex, br, y0, h, k in rows:
            if br is None:
                qp.setPen(ink)
                qp.drawText(QtCore.QRectF(4, y0, W, h), Qt.AlignVCenter | Qt.AlignLeft, tex)
                continue
            qp.save()
            qp.translate(4 - br.left() * k, y0 - br.top() * k)
            qp.scale(k, k)
            qp.fillPath(path, ink)
            qp.restore()
        qp.end()


# ----------------------------------------------------------------------------- inputs
def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    return f"{v:.3g}" if 1e-3 <= a < 1e4 else f"{v:.2e}"


class LogSlider(QtWidgets.QWidget):
    """Slider on a log (or linear) scale with an editable value box. Emits valueChanged(float) while dragging."""
    valueChanged = QtCore.pyqtSignal(float)
    STEPS = 400

    def __init__(self, value, lo, hi, log=True, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.log = float(lo), float(hi), log
        if log:
            self.lo = max(self.lo, 1e-12)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.edit = QtWidgets.QLineEdit()
        fm = self.edit.fontMetrics()
        self.edit.setMinimumWidth(fm.horizontalAdvance("-8.88e-888") + 22)
        self.edit.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.edit.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.edit)
        self._v = float(value)
        self._sync()
        self.slider.valueChanged.connect(self._from_slider)
        self.edit.editingFinished.connect(self._from_edit)

    def _to_pos(self, v):
        if self.log:
            v = min(max(v, self.lo), self.hi)
            return int(round((np.log10(v) - np.log10(self.lo)) / (np.log10(self.hi) - np.log10(self.lo)) * self.STEPS))
        return int(round((min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * self.STEPS))

    def _from_pos(self, p):
        f = p / self.STEPS
        if self.log:
            return float(10 ** (np.log10(self.lo) + f * (np.log10(self.hi) - np.log10(self.lo))))
        return float(self.lo + f * (self.hi - self.lo))

    def _sync(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(self._v))
        self.slider.blockSignals(False)
        self.edit.setText(fmt(self._v))

    def _from_slider(self, p):
        self._v = self._from_pos(p)
        v = float(f"{self._v:.3g}")
        self._v = v
        self.edit.setText(fmt(v))
        self.valueChanged.emit(v)

    def _from_edit(self):
        try:
            v = float(self.edit.text())
        except ValueError:
            self._sync()
            return
        if self.log and v <= 0:
            v = self.lo
        if v != self._v:
            self._v = v
            self._sync()
            self.valueChanged.emit(v)

    def value(self):
        return self._v

    def setValue(self, v, emit=False):
        self._v = float(v)
        self._sync()
        if emit:
            self.valueChanged.emit(self._v)


class ParamForm(QtWidgets.QWidget):
    """Form generated from a method's Param list. Emits changed(key) on every edit."""
    changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = QtWidgets.QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.widgets: dict = {}
        self.specs: dict = {}

    def build(self, params, values):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.widgets, self.specs = {}, {}
        for p in params:
            v = values.get(p.key, p.default)
            if p.kind == "float":
                w = LogSlider(v, p.lo, p.hi, log=p.log)
                w.valueChanged.connect(lambda _=None, k=p.key: self.changed.emit(k))
            elif p.kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(int(p.lo), int(p.hi))
                w.setValue(int(v))
                w.setKeyboardTracking(False)
                w.setAccelerated(True)
                w.valueChanged.connect(lambda _=None, k=p.key: self.changed.emit(k))
            elif p.kind == "choice":
                w = QtWidgets.QComboBox()
                w.addItems(list(p.choices))
                w.setCurrentText(str(v))
                w.currentTextChanged.connect(lambda _=None, k=p.key: self.changed.emit(k))
            else:
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(v))
                w.toggled.connect(lambda _=None, k=p.key: self.changed.emit(k))
            if p.tip:
                w.setToolTip(p.tip)
            lbl = QtWidgets.QLabel("λ" if p.key == "lam" and p.label == "lambda" else p.label)
            if p.tip:
                lbl.setToolTip(p.tip)
            self.form.addRow(lbl, w)
            self.widgets[p.key] = w
            self.specs[p.key] = p

    def values(self):
        out = {}
        for k, w in self.widgets.items():
            p = self.specs[k]
            if p.kind == "float":
                out[k] = w.value()
            elif p.kind == "int":
                out[k] = int(w.value())
            elif p.kind == "choice":
                out[k] = w.currentText()
            else:
                out[k] = w.isChecked()
        return out

    def set_value(self, key, v, emit=False):
        w = self.widgets.get(key)
        if w is None:
            return
        p = self.specs[key]
        w.blockSignals(True)
        if p.kind == "float":
            w.setValue(v)
        elif p.kind == "int":
            w.setValue(int(v))
        elif p.kind == "choice":
            w.setCurrentText(str(v))
        else:
            w.setChecked(bool(v))
        w.blockSignals(False)
        if emit:
            self.changed.emit(key)


def card(title):
    """Rounded panel with a title; returns (frame, body layout)."""
    f = QtWidgets.QFrame()
    f.setObjectName("card")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(8)
    t = QtWidgets.QLabel(title)
    t.setObjectName("cardTitle")
    lay.addWidget(t)
    return f, lay


def muted(text=""):
    l = QtWidgets.QLabel(text)
    l.setObjectName("muted")
    l.setWordWrap(True)
    return l


class GroupedCombo(QtWidgets.QComboBox):
    """Combo box with non-selectable group headers and per-item disabled state + tooltip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(QtGui.QStandardItemModel(self))
        self.setMaxVisibleItems(30)

    def set_groups(self, groups, labels, disabled=None):
        disabled = disabled or {}
        m = self.model()
        self.blockSignals(True)
        m.clear()
        for g, keys in groups.items():
            h = QtGui.QStandardItem(g.upper())
            h.setFlags(Qt.NoItemFlags)
            f = h.font()
            f.setBold(True)
            f.setPointSizeF(f.pointSizeF() * 0.85)
            h.setFont(f)
            m.appendRow(h)
            for k in keys:
                it = QtGui.QStandardItem("   " + labels[k])
                it.setData(k, Qt.UserRole)
                why = disabled.get(k, "")
                if why:
                    it.setFlags(Qt.NoItemFlags)
                    it.setToolTip(why)
                m.appendRow(it)
        self.blockSignals(False)

    def current_key(self):
        return self.currentData(Qt.UserRole)

    def set_key(self, key):
        for i in range(self.count()):
            if self.itemData(i, Qt.UserRole) == key:
                self.setCurrentIndex(i)
                return True
        return False

    def first_enabled(self):
        m = self.model()
        for i in range(self.count()):
            if m.item(i).flags() & Qt.ItemIsEnabled:
                return self.itemData(i, Qt.UserRole)
        return None

    def is_enabled(self, key):
        m = self.model()
        for i in range(self.count()):
            if self.itemData(i, Qt.UserRole) == key:
                return bool(m.item(i).flags() & Qt.ItemIsEnabled)
        return False


# ----------------------------------------------------------------------------- worker
class Worker(QtCore.QThread):
    """Runs fn(progress_emit, stop_flag) in a thread; cooperative cancellation via a threading.Event."""
    progress = QtCore.pyqtSignal(object)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, fn, tag=""):
        super().__init__()
        self.fn = fn
        self.tag = tag
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        from invtoolbox.core.methods import Stopped
        try:
            out = self.fn(self.progress.emit, self.stop_event.is_set)
            if self.stop_event.is_set():
                self.cancelled.emit()
            else:
                self.finished_ok.emit(out)
        except Stopped:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}")
