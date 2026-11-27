"""Reconstruction methods.

Every method has the signature ``fn(problem, data, ctx, **params) -> Result`` and is registered with a
parameter spec (for the GUI / CLI), equations (rendered in the GUI) and an applicability test.
Convention for variational methods:  minimize  D(Ax, b) + lam * R(x)  with D = 0.5||Ax - b||^2 unless
the method offers another data term. Operators are normalized to ||A|| = 1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import prox as P
from .operators import Grad, SymGrad
from .spectral import Spectral
from .wavelets import Wavelet


class Stopped(Exception):
    pass


# ----------------------------------------------------------------------------- infrastructure
@dataclass
class Param:
    key: str
    label: str
    default: object
    lo: float | None = None
    hi: float | None = None
    log: bool = False
    kind: str = "float"           # float | int | choice | bool
    choices: tuple = ()
    tip: str = ""


@dataclass
class Result:
    method: str
    x: np.ndarray
    hist: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    std: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    seconds: float = 0.0
    params: dict = field(default_factory=dict)


class Ctx:
    """Bookkeeping passed to every solver: iteration history, live callback, cooperative stop."""

    def __init__(self, problem, data, callback=None, stop=None, every=0.05):
        self.problem, self.data = problem, data
        self.callback, self.stop = callback, stop
        self.every = every
        self._last = 0.0
        self.hist = {"obj": [], "res": [], "err": []}
        xt = problem.x_true
        self._nt = max(np.linalg.norm(xt), 1e-300)

    def record(self, k, x, obj=None, res=None):
        p = self.problem
        xa = p.align(x)
        self.hist["err"].append(float(np.linalg.norm(xa - p.x_true) / self._nt))
        if res is None:
            res = float(np.linalg.norm(p.forward(x) - self.data.b))
        self.hist["res"].append(float(res))
        self.hist["obj"].append(float(obj) if obj is not None else np.nan)
        if self.stop is not None and self.stop():
            raise Stopped
        if self.callback is not None:
            now = time.perf_counter()
            if now - self._last >= self.every:
                self._last = now
                self.callback(k, x, self.hist)

    def check(self):
        if self.stop is not None and self.stop():
            raise Stopped


@dataclass
class Method:
    key: str
    label: str
    group: str
    fn: callable
    params: list
    equations: list
    blurb: str
    applies: callable
    lam_power: float = 1.0        # default lam scales like (noise level)^lam_power
    iterative: bool = True
    bayes: bool = False

    def defaults(self):
        return {p.key: p.default for p in self.params}


METHODS: dict[str, Method] = {}


def method(key, label, group, params, equations, blurb, applies, **kw):
    def deco(fn):
        METHODS[key] = Method(key, label, group, fn, params, equations, blurb, applies, **kw)
        return fn
    return deco


# applicability predicates: return "" if OK, else the reason
def _linear(p):
    return "" if p.linear else "needs a linear forward operator"


def _nonlinear(p):
    return "" if not p.linear else "for nonlinear problems"


def _phase(p):
    return "" if p.key == "phase-retrieval" else "phase retrieval only"


def _nonneg_op(p):
    if not p.linear:
        return "needs a linear forward operator"
    return "" if p.A.nonneg else "needs a nonnegative operator"


def _explicit(p):
    if not p.linear:
        return "needs a linear forward operator"
    return "" if p.A.matrix is not None else "needs explicit matrix rows"


def _spectral(p):
    if not p.linear:
        return "needs a linear forward operator"
    return "" if Spectral.available(p) else "needs an SVD (small matrix) or a convolution operator"


def _dense_1d(p):
    if not p.linear:
        return "needs a linear forward operator"
    return "" if p.category == "Signals" else "sparse-signal problems only"


def _ct(p):
    return "" if p.data_kind == "sinogram" else "CT (sinogram) problems only"


def _any(p):
    return ""


LAM = lambda d=1e-2, lo=1e-6, hi=1e2: Param("lam", "lambda", d, lo, hi, log=True,
                                             tip="regularization weight")
ITERS = lambda d=200, hi=5000: Param("iters", "iterations", d, 1, hi, kind="int")
NONNEG = lambda d=True: Param("nonneg", "x >= 0", d, kind="bool", tip="project onto the nonnegative orthant")
PENALTY = Param("penalty", "penalty L", "identity", kind="choice", choices=("identity", "gradient", "laplacian"),
                tip="R(x) = 0.5||L x||^2")
DISC = Param("discrepancy", "stop at discrepancy", False, kind="bool",
             tip="Morozov: stop once ||Ax - b|| <= tau * delta")
TAU = Param("tau", "tau", 1.02, 1.0, 3.0, tip="discrepancy safety factor")


def _penalty_ops(problem, kind):
    """Return (L, L^T) callables for the quadratic penalty 0.5 ||L x||^2."""
    if kind == "identity":
        return (lambda x: x), (lambda y: y)
    D = Grad(problem.shape)
    if kind == "gradient":
        return (lambda x: D(x).ravel()), (lambda y: D.T(y))
    if kind == "laplacian":
        return (lambda x: D.T(D(x).ravel())), (lambda y: D.T(D(y).ravel()))
    raise ValueError(kind)


def _nonneg_ok(problem, flag):
    return bool(flag) and problem.nonneg


def _finish(name, x, ctx, t0, params, **info):
    return Result(name, x, {k: np.asarray(v) for k, v in ctx.hist.items()}, info, seconds=time.perf_counter() - t0,
                  params=params)


# ============================================================================= direct / spectral
@method("adjoint", "Back-projection (scaled A^T b)", "Direct", [], [r"\hat{x} = c\,A^{T}b,\qquad c = \frac{\langle AA^{T}b,\ b\rangle}{\|AA^{T}b\|^2}"],
        "Matched filter / zero-filling baseline.", _linear, iterative=False)
def m_adjoint(problem, data, ctx, **p):
    t0 = time.perf_counter()
    x = problem.A.T(data.b)
    Ax = problem.A(x)
    x *= float(Ax @ data.b) / max(float(Ax @ Ax), 1e-300)
    ctx.record(0, x)
    return _finish("adjoint", x, ctx, t0, p)


@method("fbp", "Filtered back-projection", "Direct",
        [Param("window", "window", "hann", kind="choice", choices=("ram-lak", "shepp-logan", "cosine", "hann"))],
        [r"\hat{x} = c\,\mathcal{R}^{T}\left(|\omega|\,w(\omega) \ast_s b\right)"],
        "Classical analytic CT inversion with an apodized ramp filter.", _ct, iterative=False)
def m_fbp(problem, data, ctx, window="hann", **p):
    t0 = time.perf_counter()
    na, nd = problem.data_shape
    sino = data.b.reshape(na, nd)
    L = 1 << int(np.ceil(np.log2(2 * nd)))
    w = np.fft.fftfreq(L)
    ramp = np.abs(w)
    wn = w / 0.5
    win = {"ram-lak": np.ones_like(w), "shepp-logan": np.sinc(wn / 2), "cosine": np.cos(np.pi * wn / 2),
           "hann": 0.5 * (1 + np.cos(np.pi * wn))}[window]
    F = np.fft.fft(sino, n=L, axis=1) * (ramp * win)
    filt = np.fft.ifft(F, axis=1).real[:, :nd]
    x = problem.A.T(filt.ravel())
    Ax = problem.A(x)
    x *= float(Ax @ data.b) / max(float(Ax @ Ax), 1e-300)
    ctx.record(0, x)
    return _finish("fbp", x, ctx, t0, dict(window=window))


@method("tsvd", "Truncated SVD", "Direct",
        [Param("k", "rank k (0 = discrepancy)", 0, 0, 100000, kind="int", tip="number of retained components")],
        [r"\hat{x}_k = \sum_{i \leq k} \frac{u_i^{T}b}{\sigma_i}\,v_i"],
        "Spectral cut-off; k chosen by the discrepancy principle when 0.", _spectral, iterative=False)
def m_tsvd(problem, data, ctx, k=0, **p):
    t0 = time.perf_counter()
    S = Spectral(problem)
    beta = S.beta(data.b)
    o = np.argsort(-S.s)
    ab2 = np.abs(beta[o]) ** 2
    oor = S.out_of_range_sq(data.b, beta)
    # residual^2(k) = oor + sum_{i>k} |beta_i|^2
    tail = np.concatenate([np.cumsum(ab2[::-1])[::-1], [0.0]]) + oor
    k_disc = int(np.argmax(np.sqrt(tail) <= 1.02 * data.delta)) if np.any(np.sqrt(tail) <= 1.02 * data.delta) else len(o)
    kk = int(k) if k and k > 0 else max(k_disc, 1)
    kk = min(kk, int(np.sum(S.s > 1e-14 * S.s.max())))
    x = S.solve(data.b, S.tsvd(S.s, kk), beta)
    ctx.record(0, x)
    return _finish("tsvd", x, ctx, t0, dict(k=k), k=kk, k_discrepancy=k_disc)


def _fourier_penalty_symbol(shape, kind):
    if kind == "identity":
        return np.ones(shape)
    ws = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(s) for s in shape], indexing="ij")
    lap = sum(2 - 2 * np.cos(w) for w in ws)
    return lap if kind == "gradient" else lap ** 2


def tikhonov_solve(problem, b, lam, penalty="identity", x0=None, cg_iters=300, ctx=None):
    """argmin 0.5||Ax - b||^2 + 0.5 lam ||L x||^2, choosing the fastest exact path."""
    A = problem.A
    if A.fourier is not None:
        H, shape = A.fourier
        G = _fourier_penalty_symbol(shape, penalty)
        B = np.fft.fftn(b.reshape(shape))
        return np.fft.ifftn(np.conj(H) * B / (np.abs(H) ** 2 + lam * G + 1e-300)).real.ravel(), "fft"
    if penalty == "identity" and A.has_svd:
        S = Spectral(problem)
        return S.solve(b, S.tikhonov(S.s, lam)), "svd"
    if A.matrix is not None and problem.n <= 1100:
        Ad = A.dense()
        L, _ = _penalty_ops(problem, penalty)
        if penalty == "identity":
            LtL = np.eye(problem.n)
        else:
            Lm = np.column_stack([L(e) for e in np.eye(problem.n)])
            LtL = Lm.T @ Lm
        M = Ad.T @ Ad + lam * LtL
        return sla.solve(M, Ad.T @ b, assume_a="sym"), "dense"
    L, Lt = _penalty_ops(problem, penalty)
    x = P.cg(lambda v: A.T(A(v)) + lam * Lt(L(v)), A.T(b), x0=x0, n_iter=cg_iters, tol=1e-7)
    return x, "cg"


@method("tikhonov", "Tikhonov (generalized)", "Direct", [LAM(1e-3), PENALTY],
        [r"\hat{x} = \arg\min_x\ \frac{1}{2}\|Ax - b\|_2^2 + \frac{\lambda}{2}\|Lx\|_2^2",
         r"(A^{T}A + \lambda L^{T}L)\,\hat{x} = A^{T}b"],
        "Closed form via FFT or SVD when available, else dense solve / CG.", _linear, lam_power=2.0, iterative=False)
def m_tikhonov(problem, data, ctx, lam=1e-3, penalty="identity", **p):
    t0 = time.perf_counter()
    x, path = tikhonov_solve(problem, data.b, lam, penalty)
    ctx.record(0, x)
    return _finish("tikhonov", x, ctx, t0, dict(lam=lam, penalty=penalty), path=path)


# ============================================================================= iterative regularization
@method("landweber", "Landweber", "Iterative regularization", [ITERS(300), Param("omega", "step (x 1/||A||^2)", 1.0, 0.05, 1.99), NONNEG(False), DISC, TAU],
        [r"x_{k+1} = P_{\mathcal{C}}\left(x_k + \omega\,A^{T}(b - Ax_k)\right)", r"\mathrm{regularization\ parameter} = k"],
        "Gradient descent on the residual; early stopping regularizes (semiconvergence).", _linear)
def m_landweber(problem, data, ctx, iters=300, omega=1.0, nonneg=False, discrepancy=False, tau=1.02, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    x = np.zeros(problem.n)
    nn = _nonneg_ok(problem, nonneg)
    for k in range(int(iters)):
        r = b - A(x)
        res = float(np.linalg.norm(r))
        if discrepancy and res <= tau * data.delta:
            break
        x = x + omega * A.T(r)
        if nn:
            np.maximum(x, 0, out=x)
        ctx.record(k, x, obj=0.5 * res ** 2)
    return _finish("landweber", x, ctx, t0, dict(iters=iters, omega=omega, nonneg=nonneg), stopped_at=len(ctx.hist["err"]))


@method("cgls", "CGLS (Krylov)", "Iterative regularization", [ITERS(50, 2000), DISC, TAU],
        [r"x_k = \arg\min_{x \in \mathcal{K}_k(A^{T}A,\ A^{T}b)} \|Ax - b\|_2"],
        "Conjugate gradients on the normal equations; the iteration count is the regularization parameter.", _linear)
def m_cgls(problem, data, ctx, iters=50, discrepancy=False, tau=1.02, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    x = np.zeros(problem.n)
    r = b.copy()
    s = A.T(r)
    pdir = s.copy()
    gamma = float(s @ s)
    for k in range(int(iters)):
        q = A(pdir)
        alpha = gamma / max(float(q @ q), 1e-300)
        x = x + alpha * pdir
        r = r - alpha * q
        res = float(np.linalg.norm(r))
        ctx.record(k, x, obj=0.5 * res ** 2, res=res)
        if discrepancy and res <= tau * data.delta:
            break
        s = A.T(r)
        g_new = float(s @ s)
        if g_new < 1e-28:
            break
        pdir = s + (g_new / gamma) * pdir
        gamma = g_new
    return _finish("cgls", x, ctx, t0, dict(iters=iters), stopped_at=len(ctx.hist["err"]))


@method("sirt", "SIRT", "Iterative regularization", [ITERS(200), NONNEG(True), DISC, TAU],
        [r"x_{k+1} = P_{\mathcal{C}}\left(x_k + C A^{T} R\,(b - Ax_k)\right)",
         r"R = \mathrm{diag}(1/A\mathbf{1}),\quad C = \mathrm{diag}(1/A^{T}\mathbf{1})"],
        "Row/column-sum preconditioned Landweber (CT standard).", _nonneg_op)
def m_sirt(problem, data, ctx, iters=200, nonneg=True, discrepancy=False, tau=1.02, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    rs = A.row_sums()
    cs = A.col_sums()
    R = np.where(rs > 1e-12, 1 / np.maximum(rs, 1e-12), 0)
    C = np.where(cs > 1e-12, 1 / np.maximum(cs, 1e-12), 0)
    nn = _nonneg_ok(problem, nonneg)
    x = np.zeros(problem.n)
    for k in range(int(iters)):
        r = b - A(x)
        res = float(np.linalg.norm(r))
        if discrepancy and res <= tau * data.delta:
            break
        x = x + C * A.T(R * r)
        if nn:
            np.maximum(x, 0, out=x)
        ctx.record(k, x, obj=0.5 * res ** 2)
    return _finish("sirt", x, ctx, t0, dict(iters=iters, nonneg=nonneg))


@method("kaczmarz", "Kaczmarz / ART", "Iterative regularization",
        [ITERS(10, 500), Param("omega", "relaxation", 0.25, 0.01, 1.99), NONNEG(True),
         Param("random", "randomized order", True, kind="bool")],
        [r"x \leftarrow x + \omega\,\frac{b_i - a_i^{T}x}{\|a_i\|^2}\,a_i,\qquad i = 1..m\ \mathrm{per\ sweep}"],
        "Row-action projections; one iteration = one sweep over all rows.", _explicit)
def m_kaczmarz(problem, data, ctx, iters=10, omega=0.25, nonneg=True, random=True, **p):
    t0 = time.perf_counter()
    M = problem.A.matrix
    M = M.tocsr() if sp.issparse(M) else sp.csr_matrix(M)
    b = data.b
    indptr, indices, vals = M.indptr, M.indices, M.data
    rn = np.asarray(M.multiply(M).sum(axis=1)).ravel()
    rows = np.flatnonzero(rn > 1e-14)
    rng = np.random.default_rng(0)
    nn = _nonneg_ok(problem, nonneg)
    x = np.zeros(problem.n)
    for k in range(int(iters)):
        order = rng.permutation(rows) if random else rows
        for i in order:
            a, z = indptr[i], indptr[i + 1]
            idx, v = indices[a:z], vals[a:z]
            x[idx] += (omega * (b[i] - v @ x[idx]) / rn[i]) * v
        if nn:
            np.maximum(x, 0, out=x)
        ctx.record(k, x)
    return _finish("kaczmarz", x, ctx, t0, dict(iters=iters, omega=omega))


# ============================================================================= sparsity
BASIS = Param("basis", "sparsity basis", "pixel", kind="choice", choices=("pixel", "haar", "db2", "db4"),
              tip="l1 on pixels or on orthonormal wavelet detail coefficients")


def _l1_prox_factory(problem, basis, nonneg):
    if basis == "pixel":
        if nonneg:
            return lambda v, t: np.maximum(v - t, 0.0), lambda x: float(np.abs(x).sum())
        return lambda v, t: P.soft(v, t), lambda x: float(np.abs(x).sum())
    W = Wavelet(problem.shape, basis)
    mask = W.detail_mask()

    def prox(v, t):
        c = W.forward(v)
        c[mask] = P.soft(c[mask], t)
        x = W.inverse(c)
        return np.maximum(x, 0) if nonneg else x

    return prox, (lambda x: float(np.abs(W.forward(x)[mask]).sum()))


def _ista(problem, data, ctx, lam, iters, nonneg, basis, accelerated, restart, name):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    L = A.norm() ** 2
    step = 1.0 / L
    proxf, R = _l1_prox_factory(problem, basis, _nonneg_ok(problem, nonneg))
    x = np.zeros(problem.n)
    z = x.copy()
    t = 1.0
    for k in range(int(iters)):
        rz = A(z) - b
        x_new = proxf(z - step * A.T(rz), lam * step)
        if accelerated:
            if restart and float((z - x_new) @ (x_new - x)) > 0:
                t = 1.0
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = x_new + (t - 1) / t_new * (x_new - x)
            t = t_new
        else:
            z = x_new
        x = x_new
        r = A(x) - b
        res = float(np.linalg.norm(r))
        ctx.record(k, x, obj=0.5 * res ** 2 + lam * R(x), res=res)
    return _finish(name, x, ctx, t0, dict(lam=lam, iters=iters, nonneg=nonneg, basis=basis))


@method("ista", "ISTA", "Sparsity", [LAM(1e-3), ITERS(300), BASIS, NONNEG(False)],
        [r"\min_x\ \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|\Psi x\|_1",
         r"x_{k+1} = \mathrm{prox}_{\lambda t\|\Psi\cdot\|_1}\left(x_k - t A^{T}(Ax_k - b)\right)"],
        "Proximal gradient; O(1/k) in objective.", _linear)
def m_ista(problem, data, ctx, lam=1e-3, iters=300, basis="pixel", nonneg=False, **p):
    return _ista(problem, data, ctx, lam, iters, nonneg, basis, False, False, "ista")


@method("fista", "FISTA (+ adaptive restart)", "Sparsity",
        [LAM(1e-3), ITERS(300), BASIS, NONNEG(False), Param("restart", "adaptive restart", True, kind="bool")],
        [r"\min_x\ \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|\Psi x\|_1",
         r"z_{k+1} = x_{k+1} + \frac{t_k - 1}{t_{k+1}}(x_{k+1} - x_k)"],
        "Nesterov-accelerated proximal gradient, O(1/k^2), gradient-based restart.", _linear)
def m_fista(problem, data, ctx, lam=1e-3, iters=300, basis="pixel", nonneg=False, restart=True, **p):
    return _ista(problem, data, ctx, lam, iters, nonneg, basis, True, restart, "fista")


@method("admm-l1", "ADMM (lasso)", "Sparsity", [LAM(1e-3), ITERS(100), Param("rho", "rho", 1.0, 1e-3, 1e3, log=True)],
        [r"\min_{x,z}\ \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|z\|_1\quad \mathrm{s.t.}\ x = z"],
        "Augmented Lagrangian splitting; x-update by CG.", _linear)
def m_admm_l1(problem, data, ctx, lam=1e-3, iters=100, rho=1.0, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    Atb = A.T(b)
    x = np.zeros(problem.n)
    z = x.copy()
    u = x.copy()
    for k in range(int(iters)):
        x = P.cg(lambda v: A.T(A(v)) + rho * v, Atb + rho * (z - u), x0=x, n_iter=10)
        z = P.soft(x + u, lam / rho)
        u += x - z
        res = float(np.linalg.norm(A(z) - b))
        ctx.record(k, z, obj=0.5 * res ** 2 + lam * np.abs(z).sum(), res=res)
    return _finish("admm-l1", z, ctx, t0, dict(lam=lam, iters=iters, rho=rho))


@method("omp", "Orthogonal matching pursuit", "Sparsity",
        [Param("k", "max atoms", 40, 1, 1000, kind="int"), DISC, TAU],
        [r"\min_x\ \|Ax - b\|_2\quad \mathrm{s.t.}\ \|x\|_0 \leq k",
         r"j^\ast = \arg\max_j |a_j^{T} r| / \|a_j\|"],
        "Greedy support selection with least-squares refits.", _dense_1d)
def m_omp(problem, data, ctx, k=40, discrepancy=True, tau=1.02, **p):
    t0 = time.perf_counter()
    Ad = problem.A.dense()
    b = data.b
    nrm = np.linalg.norm(Ad, axis=0) + 1e-300
    r = b.copy()
    S: list[int] = []
    x = np.zeros(problem.n)
    for it in range(int(min(k, problem.A.m))):
        j = int(np.argmax(np.abs(Ad.T @ r) / nrm))
        if j in S:
            break
        S.append(j)
        coef, *_ = np.linalg.lstsq(Ad[:, S], b, rcond=None)
        x = np.zeros(problem.n)
        x[S] = coef
        r = b - Ad @ x
        res = float(np.linalg.norm(r))
        ctx.record(it, x, obj=0.5 * res ** 2, res=res)
        if discrepancy and res <= tau * data.delta:
            break
    return _finish("omp", x, ctx, t0, dict(k=k), support=len(S))


@method("irls", "IRLS (lp, p < 1)", "Sparsity",
        [LAM(1e-3), Param("p", "p", 0.5, 0.05, 1.0), ITERS(30, 300),
         Param("operator", "sparsify", "identity", kind="choice", choices=("identity", "gradient"))],
        [r"\min_x\ \frac{1}{2}\|Ax - b\|_2^2 + \lambda \sum_i \left(|(Lx)_i|^2 + \epsilon^2\right)^{p/2}",
         r"(A^{T}A + \lambda p\,L^{T}W_kL)\,x_{k+1} = A^{T}b,\quad W_k = \mathrm{diag}(|Lx_k|^2 + \epsilon_k^2)^{p/2 - 1}"],
        "Nonconvex lp quasi-norm by iteratively reweighted least squares with epsilon continuation.", _linear)
def m_irls(problem, data, ctx, lam=1e-3, p=0.5, iters=30, operator="identity", **kw):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    L, Lt = _penalty_ops(problem, "identity" if operator == "identity" else "gradient")
    Atb = A.T(b)
    x, _ = tikhonov_solve(problem, b, 1e-3, "identity" if operator == "identity" else "gradient")
    eps = max(np.abs(L(x)).max(), 1e-3)
    for k in range(int(iters)):
        w = (L(x) ** 2 + eps ** 2) ** (p / 2 - 1)
        x = P.cg(lambda v: A.T(A(v)) + lam * p * Lt(w * L(v)), Atb, x0=x, n_iter=60, tol=1e-8)
        eps = max(eps * 0.6, 1e-6)
        r = A(x) - b
        res = float(np.linalg.norm(r))
        obj = 0.5 * res ** 2 + lam * float(np.sum((L(x) ** 2 + 1e-12) ** (p / 2)))
        ctx.record(k, x, obj=obj, res=res)
    return _finish("irls", x, ctx, t0, dict(lam=lam, p=p, iters=iters, operator=operator))


# ============================================================================= total variation family
ISO = Param("iso", "isotropic TV", True, kind="bool")


@method("admm-tv", "ADMM-TV", "Total variation", [LAM(1e-3), ITERS(100), Param("rho", "rho", 0.01, 1e-4, 1e2, log=True), ISO],
        [r"\min_x\ \frac{1}{2}\|Ax - b\|_2^2 + \lambda\,\|\nabla x\|_{2,1}",
         r"(A^{T}A + \rho\nabla^{T}\nabla)\,x^{+} = A^{T}b + \rho\nabla^{T}(z - u)"],
        "Split z = grad x; x-update by warm-started CG, z-update by (group) shrinkage.", _linear)
def m_admm_tv(problem, data, ctx, lam=1e-3, iters=100, rho=0.01, iso=True, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    D = Grad(problem.shape)
    Atb = A.T(b)
    x = np.zeros(problem.n)
    z = D(x)
    u = np.zeros_like(z)
    for k in range(int(iters)):
        x = P.cg(lambda v: A.T(A(v)) + rho * D.T(D(v)), Atb + rho * D.T(z - u), x0=x, n_iter=8)
        Dx = D(x)
        v = Dx + u
        if iso:
            nrm = np.sqrt((v * v).sum(0, keepdims=True))
            z = v * np.maximum(1 - (lam / rho) / np.maximum(nrm, 1e-300), 0)
        else:
            z = P.soft(v, lam / rho)
        u = v - z
        res = float(np.linalg.norm(A(x) - b))
        ctx.record(k, x, obj=0.5 * res ** 2 + lam * P.tv(x, problem.shape, iso), res=res)
    return _finish("admm-tv", x, ctx, t0, dict(lam=lam, iters=iters, rho=rho, iso=iso))


def _data_prox(kind, b, scale):
    if kind == "L2":
        return lambda v, s: (v - s * b) / (1 + s)
    if kind == "L1":
        return lambda v, s: np.clip(v - s * b, -1, 1)
    if kind == "KL":
        bb = np.maximum(b, 0)
        return lambda v, s: 0.5 * (1 + v - np.sqrt((v - 1) ** 2 + 4 * s * bb))
    raise ValueError(kind)


def _data_value(kind, y, b):
    if kind == "L2":
        return 0.5 * float(np.sum((y - b) ** 2))
    if kind == "L1":
        return float(np.abs(y - b).sum())
    yy = np.maximum(y, 1e-12)
    bb = np.maximum(b, 0)
    return float(np.sum(yy - bb + bb * np.log(np.where(bb > 0, bb, 1) / yy)))


@method("pdhg", "Primal-dual (Chambolle-Pock)", "Total variation",
        [LAM(1e-3), ITERS(300),
         Param("data_term", "data term", "L2", kind="choice", choices=("L2", "L1", "KL"),
               tip="L2 Gaussian, L1 impulsive, KL Poisson"),
         Param("reg", "regularizer", "TV", kind="choice", choices=("TV", "TGV2", "L1", "wavelet")),
         Param("alpha", "TGV alpha0/alpha1", 2.0, 0.1, 20, log=True), NONNEG(True),
         Param("ratio", "step ratio tau/sigma", 20.0, 0.1, 1000, log=True,
               tip="primal/dual step balance (tau * sigma * ||K||^2 < 1 always holds)")],
        [r"\min_x\ \mathcal{D}(Ax, b) + \lambda\,\mathcal{R}(x) + \iota_{x\geq 0}",
         r"\mathrm{TGV}^2(x) = \min_w\ \|\nabla x - w\|_{2,1} + \alpha\,\|\mathcal{E}w\|_{2,1}",
         r"\mathcal{D} \in \{\frac{1}{2}\|\cdot\|_2^2,\ \|\cdot\|_1,\ \mathrm{KL}\}"],
        "One saddle-point solver for L2/L1/KL data terms and TV, TGV, l1, wavelet-l1 priors.", _linear)
def m_pdhg(problem, data, ctx, lam=1e-3, iters=300, data_term="L2", reg="TV", alpha=2.0, nonneg=True, ratio=20.0, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    if data_term == "KL" and not A.nonneg:
        raise ValueError("KL data term needs a nonnegative operator")
    shape = problem.shape
    D = Grad(shape)
    E = SymGrad(shape)
    W = Wavelet(shape, "db2") if reg == "wavelet" else None
    tgv = reg == "TGV2"
    nn = _nonneg_ok(problem, nonneg) or data_term == "KL"
    pd = _data_prox(data_term, b, data.scale)
    nd = D.ndim
    # K = [A, 0; R1, -I_w; 0, E] ; ||K||^2 <= ||A||^2 + ||R1||^2 (+ 1 + ||E||^2 for TGV)
    if reg == "TV" or tgv:
        R1, R1T, rn2 = (lambda x: D(x)), (lambda q: D.T(q)), D.norm_sq()
    elif reg == "L1":
        R1, R1T, rn2 = (lambda x: x.copy()), (lambda q: q.ravel()), 1.0
    else:
        R1, R1T, rn2 = (lambda x: W.forward(x)), (lambda q: W.inverse(q)), 1.0
    Knorm = np.sqrt(1.0 + rn2 + (1.0 + E.norm_sq() if tgv else 0.0))
    tau, sigma = 0.99 * ratio / Knorm, 0.99 / (ratio * Knorm)
    x = np.zeros(problem.n) if data_term != "KL" else np.full(problem.n, max(b.mean(), 1e-3))
    w = np.zeros((nd,) + shape) if tgv else None
    xb, wb = x.copy(), (w.copy() if tgv else None)
    y1 = np.zeros(A.m)
    y2 = np.zeros_like(R1(x))
    y3 = np.zeros((E.nc,) + shape) if tgv else None
    mask = W.detail_mask() if W is not None else None
    iso = reg in ("TV", "TGV2")
    for k in range(int(iters)):
        y1 = pd(y1 + sigma * A(xb), sigma)
        g = R1(xb) - (wb if tgv else 0)
        if iso:
            y2 = P.project_ball(y2 + sigma * g, lam, True)
        else:
            y2 = np.clip(y2 + sigma * g, -lam, lam)
            if mask is not None:
                y2[~mask] = 0.0
        if tgv:
            y3 = P.project_ball(y3 + sigma * E(wb), lam * alpha, True)
        x_new = x - tau * (A.T(y1) + R1T(y2))
        if nn:
            np.maximum(x_new, 0, out=x_new)
        if tgv:
            w_new = w - tau * (-y2 + E.T(y3))
            wb = 2 * w_new - w
            w = w_new
        xb = 2 * x_new - x
        x = x_new
        Ax = A(x)
        res = float(np.linalg.norm(Ax - b))
        if k % 5 == 0 or k == iters - 1:
            if reg == "TV":
                rv = P.tv(x, shape)
            elif tgv:
                gw = D(x) - w
                rv = float(np.sqrt((gw ** 2).sum(0)).sum() + alpha * np.sqrt((E(w) ** 2).sum(0)).sum())
            elif reg == "L1":
                rv = float(np.abs(x).sum())
            else:
                rv = float(np.abs(W.forward(x)[mask]).sum())
            obj = _data_value(data_term, Ax, b) + lam * rv
        else:
            obj = np.nan
        ctx.record(k, x, obj=obj, res=res)
    return _finish("pdhg", x, ctx, t0, dict(lam=lam, iters=iters, data_term=data_term, reg=reg, alpha=alpha,
                                            nonneg=nonneg, ratio=ratio))


@method("richardson-lucy", "Richardson-Lucy (+TV)", "Poisson / EM",
        [ITERS(100, 3000), Param("lam", "lambda (TV)", 0.0, 0.0, 0.1, tip="0 = plain EM")],
        [r"x_{k+1} = \frac{x_k}{A^{T}\mathbf{1} - \lambda\,\mathrm{div}\left(\nabla x_k / |\nabla x_k|\right)} \odot A^{T}\frac{b}{Ax_k}"],
        "EM for the Poisson likelihood; multiplicative, positivity preserving. Iteration count regularizes.",
        _nonneg_op)
def m_rl(problem, data, ctx, iters=100, lam=0.0, **p):
    t0 = time.perf_counter()
    A = problem.A
    b = np.maximum(data.b, 0)
    D = Grad(problem.shape)
    c1 = A.col_sums()
    x = np.full(problem.n, max(b.sum() / max(c1.sum(), 1e-12), 1e-6))
    for k in range(int(iters)):
        Ax = np.maximum(A(x), 1e-12)
        denom = c1.copy()
        if lam > 0:
            g = D(x)
            nrm = np.sqrt((g * g).sum(0, keepdims=True) + 1e-6)
            denom = denom + lam * D.T(g / nrm)      # -div = D^T
            denom = np.maximum(denom, 1e-3 * c1.max())
        x = x / denom * A.T(b / Ax)
        Ax = A(x)
        res = float(np.linalg.norm(Ax - data.b))
        ctx.record(k, x, obj=_data_value("KL", Ax, b), res=res)
    return _finish("richardson-lucy", x, ctx, t0, dict(iters=iters, lam=lam))


# ============================================================================= plug-and-play
@method("pnp-admm", "Plug-and-play ADMM", "Plug-and-play",
        [Param("denoiser", "denoiser", "wavelet", kind="choice", choices=("wavelet", "tv", "gaussian", "median")),
         Param("strength", "denoiser strength", 0.02, 1e-4, 0.5, log=True), Param("rho", "rho", 1.0, 1e-2, 1e2, log=True),
         ITERS(40, 500)],
        [r"x^{+} = (A^{T}A + \rho I)^{-1}\left(A^{T}b + \rho(z - u)\right)",
         r"z^{+} = \mathcal{D}_\sigma(x^{+} + u),\qquad u^{+} = u + x^{+} - z^{+}"],
        "Replaces the prox of the prior by an off-the-shelf denoiser (implicit prior).", _linear)
def m_pnp(problem, data, ctx, denoiser="wavelet", strength=0.02, rho=1.0, iters=40, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    Atb = A.T(b)
    shape = problem.shape
    if A.fourier is not None:
        H, sh = A.fourier
        Hc, H2 = np.conj(H), np.abs(H) ** 2

        def xsolve(rhs, x0):
            return np.fft.ifftn(np.fft.fftn(rhs.reshape(sh)) / (H2 + rho)).real.ravel()
    else:
        def xsolve(rhs, x0):
            return P.cg(lambda v: A.T(A(v)) + rho * v, rhs, x0=x0, n_iter=30)
    x = Atb.copy()
    z = x.copy()
    u = np.zeros_like(x)
    state = None
    for k in range(int(iters)):
        x = xsolve(Atb + rho * (z - u), x)
        z, state = P.denoise(x + u, shape, denoiser, strength, state)
        if problem.nonneg:
            np.maximum(z, 0, out=z)
        u += x - z
        res = float(np.linalg.norm(A(z) - b))
        ctx.record(k, z, obj=0.5 * res ** 2, res=res)
    return _finish("pnp-admm", z, ctx, t0, dict(denoiser=denoiser, strength=strength, rho=rho, iters=iters))


# ============================================================================= Bayesian
@method("gaussian-posterior", "Gaussian posterior (exact / RTO)", "Bayesian",
        [LAM(1e-3), PENALTY, Param("samples", "samples (large problems)", 24, 4, 400, kind="int")],
        [r"p(x\,|\,b) \propto \exp\left(-\frac{1}{2\sigma^2}\|Ax - b\|^2 - \frac{\lambda}{2\sigma^2}\|Lx\|^2\right)",
         r"\Gamma_{\mathrm{post}} = \sigma^2 (A^{T}A + \lambda L^{T}L)^{-1}"],
        "Posterior mean = Tikhonov; exact marginal std for small problems, perturb-and-optimize samples otherwise.",
        _linear, lam_power=2.0, iterative=False, bayes=True)
def m_gauss_post(problem, data, ctx, lam=1e-3, penalty="identity", samples=24, **p):
    t0 = time.perf_counter()
    A, b, s = problem.A, data.b, data.sigma
    mean, path = tikhonov_solve(problem, b, lam, penalty)
    ctx.record(0, mean)
    if (A.matrix is not None or A.has_svd) and problem.n <= 1100:
        Ad = A.dense()
        L, _ = _penalty_ops(problem, penalty)
        Lm = np.eye(problem.n) if penalty == "identity" else np.column_stack([L(e) for e in np.eye(problem.n)])
        Hm = Ad.T @ Ad + lam * (Lm.T @ Lm) + 1e-10 * np.eye(problem.n)
        cov = s ** 2 * np.linalg.inv(Hm)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        how = "exact"
    else:
        L, Lt = _penalty_ops(problem, penalty)
        rng = np.random.default_rng(data.seed + 101)
        acc = np.zeros(problem.n)
        acc2 = np.zeros(problem.n)
        M = lambda v: A.T(A(v)) + lam * Lt(L(v)) + 1e-10 * v
        for i in range(int(samples)):
            ctx.check()
            e1 = rng.normal(size=A.m)
            e2 = rng.normal(size=L(mean).size)
            rhs = A.T(b + s * e1) + np.sqrt(lam) * s * Lt(e2)
            xs = P.cg(M, rhs, x0=mean, n_iter=150, tol=1e-6)
            d = xs - mean
            acc += d
            acc2 += d * d
            if ctx.callback is not None:
                ctx.callback(i, mean, ctx.hist)
        std = np.sqrt(np.maximum(acc2 / samples - (acc / samples) ** 2, 0))
        how = f"{samples} RTO samples"
    r = _finish("gaussian-posterior", mean, ctx, t0, dict(lam=lam, penalty=penalty, samples=samples), uq=how)
    r.std, r.lo, r.hi = std, mean - 1.96 * std, mean + 1.96 * std
    return r


@method("myula-tv", "MYULA (TV prior, MCMC)", "Bayesian",
        [LAM(1e-3), Param("iters", "samples", 1500, 50, 50000, kind="int"), Param("burn", "burn-in fraction", 0.2, 0.0, 0.9)],
        [r"\pi(x) \propto \exp\left(-\frac{1}{\sigma^2}\left[\frac{1}{2}\|Ax - b\|^2 + \lambda\,\mathrm{TV}(x)\right]\right)",
         r"x_{k+1} = x_k - \gamma\nabla f(x_k) - \frac{\gamma}{\mu}\left(x_k - \mathrm{prox}_{\mu g}(x_k)\right) + \sqrt{2\gamma}\,\xi_k"],
        "Moreau-Yosida unadjusted Langevin: samples a smoothed TV posterior; mean and pixelwise std.",
        _linear, bayes=True)
def m_myula(problem, data, ctx, lam=1e-3, iters=1500, burn=0.2, **p):
    t0 = time.perf_counter()
    A, b = problem.A, data.b
    s2 = data.sigma ** 2
    Lf = A.norm() ** 2 / s2
    mu = 1.0 / Lf
    gamma = 0.98 / (Lf + 1.0 / mu)
    shape = problem.shape
    rng = np.random.default_rng(data.seed + 202)
    # warm start at the (approximate) MAP
    x = m_pdhg(problem, data, Ctx(problem, data, stop=ctx.stop), lam=lam, iters=150, reg="TV", nonneg=False).x
    nb = int(burn * iters)
    acc = np.zeros(problem.n)
    acc2 = np.zeros(problem.n)
    cnt = 0
    pstate = None
    for k in range(int(iters)):
        gf = A.T(A(x) - b) / s2
        px, pstate = P.tv_prox(x, mu * lam / s2, shape, n_iter=10, p0=pstate)
        x = x - gamma * gf - (gamma / mu) * (x - px) + np.sqrt(2 * gamma) * rng.normal(size=problem.n)
        if k >= nb:
            acc += x
            acc2 += x * x
            cnt += 1
            mean = acc / cnt
            if k % 10 == 0 or k == iters - 1:
                ctx.record(k, mean)
        else:
            ctx.check()
    mean = acc / max(cnt, 1)
    std = np.sqrt(np.maximum(acc2 / max(cnt, 1) - mean ** 2, 0))
    r = _finish("myula-tv", mean, ctx, t0, dict(lam=lam, iters=iters, burn=burn), samples=cnt, step=gamma)
    r.std, r.lo, r.hi = std, mean - 1.96 * std, mean + 1.96 * std
    return r


# ============================================================================= nonlinear
def initial_guess(problem, data):
    if problem.key == "phase-retrieval":
        A = problem.meta["A"]
        b = np.maximum(data.b, 0)
        m, n = A.shape
        Y = (A.T * b) @ A / m
        w, V = np.linalg.eigh(Y)
        v = V[:, -1]
        return v * np.sqrt(max(n * b.mean(), 1e-12))
    return problem.x0(data.b)


def _gn_penalty(problem, kind):
    if kind == "identity":
        return np.eye(problem.n)
    D = Grad(problem.shape)
    Lm = np.column_stack([D(e).ravel() for e in np.eye(problem.n)])
    if kind == "gradient":
        return Lm
    return Lm.T @ Lm


@method("gauss-newton", "Gauss-Newton / Levenberg-Marquardt", "Nonlinear",
        [LAM(1e-3), Param("penalty", "penalty L", "gradient", kind="choice", choices=("identity", "gradient", "laplacian")),
         ITERS(30, 500)],
        [r"\min_x\ \frac{1}{2}\|F(x) - b\|^2 + \frac{\lambda}{2}\|L(x - x_0)\|^2",
         r"(J_k^{T}J_k + \lambda L^{T}L + \mu_k D_k)\,\delta = -\left(J_k^{T}r_k + \lambda L^{T}L(x_k - x_0)\right)",
         r"\Gamma_{\mathrm{Laplace}} = \sigma^2 (J^{T}J + \lambda L^{T}L)^{-1}"],
        "Damped Gauss-Newton with adaptive mu; returns a Laplace (linearized) posterior std.", _nonlinear,
        lam_power=2.0)
def m_gn(problem, data, ctx, lam=1e-3, penalty="gradient", iters=30, **p):
    t0 = time.perf_counter()
    b = data.b
    Lm = _gn_penalty(problem, penalty)
    LtL = Lm.T @ Lm
    x0 = problem.x0(b)
    x = initial_guess(problem, data)
    phi = lambda z, r: 0.5 * float(r @ r) + 0.5 * lam * float((z - x0) @ LtL @ (z - x0))
    r = problem.F(x) - b
    f = phi(x, r)
    mu = 1e-3
    J = problem.J(x)
    for k in range(int(iters)):
        g = J.T @ r + lam * LtL @ (x - x0)
        Hm = J.T @ J + lam * LtL
        dg = np.diag(Hm).copy() + 1e-12
        for _ in range(12):
            step = np.linalg.solve(Hm + mu * np.diag(dg), -g)
            xn = x + step
            if not np.all(np.isfinite(xn)):
                mu *= 4
                continue
            rn = problem.F(xn) - b
            fn = phi(xn, rn)
            if np.isfinite(fn) and fn < f:
                x, r, f = xn, rn, fn
                mu = max(mu / 3, 1e-9)
                break
            mu *= 4
        else:
            break
        J = problem.J(x)
        ctx.record(k, x, obj=f, res=float(np.linalg.norm(r)))
        if np.linalg.norm(step) < 1e-9 * (1 + np.linalg.norm(x)):
            break
    if not ctx.hist["err"]:
        ctx.record(0, x, obj=f)
    Hm = J.T @ J + lam * LtL + 1e-10 * np.eye(problem.n)
    std = data.sigma * np.sqrt(np.clip(np.diag(np.linalg.inv(Hm)), 0, None))
    r_ = _finish("gauss-newton", x, ctx, t0, dict(lam=lam, penalty=penalty, iters=iters))
    xa = problem.align(x)
    r_.std, r_.lo, r_.hi = std, xa - 1.96 * std, xa + 1.96 * std
    return r_


@method("lbfgs", "L-BFGS-B (smoothness prior)", "Nonlinear",
        [LAM(1e-3), Param("penalty", "penalty L", "gradient", kind="choice", choices=("identity", "gradient", "laplacian")),
         ITERS(200, 5000), NONNEG(False)],
        [r"\min_{x \in \mathcal{C}}\ \frac{1}{2}\|F(x) - b\|^2 + \frac{\lambda}{2}\|L(x - x_0)\|^2"],
        "Quasi-Newton with box constraints; works for linear and nonlinear models (adjoint gradients).", _any,
        lam_power=2.0)
def m_lbfgs(problem, data, ctx, lam=1e-3, penalty="gradient", iters=200, nonneg=False, **p):
    from scipy.optimize import minimize
    t0 = time.perf_counter()
    b = data.b
    if problem.n > 600 and penalty != "identity":
        L, Lt = _penalty_ops(problem, penalty)
    elif penalty == "identity":
        L, Lt = (lambda z: z), (lambda z: z)
    else:
        Lm = _gn_penalty(problem, penalty)
        L, Lt = (lambda z: Lm @ z), (lambda z: Lm.T @ z)
    x0 = problem.x0(b)
    xs = initial_guess(problem, data) if not problem.linear else np.zeros(problem.n)

    def fg(z):
        r = problem.forward(z) - b
        Lz = L(z - x0)
        return 0.5 * float(r @ r) + 0.5 * lam * float(Lz @ Lz), problem.grad(z, b) + lam * Lt(Lz)

    k = [0]

    def cb(z):
        ctx.record(k[0], z, obj=fg(z)[0])
        k[0] += 1

    bounds = [(0, None)] * problem.n if _nonneg_ok(problem, nonneg) else None
    res = minimize(fg, xs, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                   options={"maxiter": int(iters), "maxcor": 20, "ftol": 1e-14, "gtol": 1e-10})
    if not ctx.hist["err"]:
        ctx.record(0, res.x)
    return _finish("lbfgs", res.x, ctx, t0, dict(lam=lam, penalty=penalty, iters=iters), message=str(res.message))


@method("wirtinger", "Wirtinger flow", "Nonlinear", [ITERS(500, 5000), Param("step", "step mu", 0.2, 0.01, 1.0)],
        [r"x_{k+1} = x_k - \frac{\mu_k}{\|x_0\|^2}\,\frac{1}{m}\sum_i \left((\tilde a_i^{T}x_k)^2 - \tilde b_i\right)(\tilde a_i^{T}x_k)\,\tilde a_i"],
        "Spectral initialization + gradient descent on the intensity loss (Candes-Li-Soltanolkotabi).", _phase)
def m_wf(problem, data, ctx, iters=500, step=0.2, **p):
    t0 = time.perf_counter()
    A = problem.meta["A"]
    m, n = A.shape
    At, bt = A * np.sqrt(n), data.b * n      # standard-normal rows
    x = initial_guess(problem, data)
    n0 = float(x @ x)
    for k in range(int(iters)):
        Ax = At @ x
        g = At.T @ ((Ax ** 2 - bt) * Ax) / m
        mu = min(1 - np.exp(-(k + 1) / 50), step)
        x = x - mu / max(n0, 1e-12) * g
        r = (A @ x) ** 2 - data.b
        ctx.record(k, x, obj=0.25 * float(np.mean((Ax ** 2 - bt) ** 2)), res=float(np.linalg.norm(r)))
    return _finish("wirtinger", x, ctx, t0, dict(iters=iters, step=step))


@method("amplitude-flow", "Amplitude flow", "Nonlinear", [ITERS(300, 5000), Param("step", "step mu", 0.6, 0.05, 1.5)],
        [r"\min_x\ \frac{1}{2m}\sum_i \left(|\tilde a_i^{T}x| - \sqrt{\tilde b_i}\right)^2",
         r"x_{k+1} = x_k - \frac{\mu}{m}\sum_i \left(\tilde a_i^{T}x_k - \sqrt{\tilde b_i}\,\mathrm{sgn}(\tilde a_i^{T}x_k)\right)\tilde a_i"],
        "Gradient descent on the amplitude loss (Wang-Giannakis-Eldar): better conditioned than intensities.", _phase)
def m_af(problem, data, ctx, iters=300, step=0.6, **p):
    t0 = time.perf_counter()
    A = problem.meta["A"]
    m, n = A.shape
    At = A * np.sqrt(n)
    sb = np.sqrt(np.maximum(data.b * n, 0))
    x = initial_guess(problem, data)
    for k in range(int(iters)):
        Ax = At @ x
        g = At.T @ (Ax - sb * np.sign(Ax)) / m
        x = x - step * g
        r = (A @ x) ** 2 - data.b
        ctx.record(k, x, obj=0.5 * float(np.mean((np.abs(Ax) - sb) ** 2)), res=float(np.linalg.norm(r)))
    return _finish("amplitude-flow", x, ctx, t0, dict(iters=iters, step=step))


# ============================================================================= public API
def applicable(problem):
    """{method key: reason} with "" for applicable methods."""
    return {k: m.applies(problem) for k, m in METHODS.items()}


def groups():
    out: dict[str, list[str]] = {}
    for k, m in METHODS.items():
        out.setdefault(m.group, []).append(k)
    return out


def run(method_key, problem, data, params=None, callback=None, stop=None, every=0.05) -> Result:
    m = METHODS[method_key]
    why = m.applies(problem)
    if why:
        raise ValueError(f"{method_key}: {why}")
    kw = m.defaults()
    kw.update(params or {})
    if "lam" in kw and (params is None or params.get("lam") is None):
        from .defaults import default_lam
        kw["lam"] = default_lam(method_key, problem, data, kw)
    ctx = Ctx(problem, data, callback=callback, stop=stop, every=every)
    res = m.fn(problem, data, ctx, **kw)
    res.params = dict(kw)
    from .metrics import summary
    res.info.update(summary(res.x, problem, data))
    return res
