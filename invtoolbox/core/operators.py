"""Matrix-free linear operators.

A ``LinOp`` maps flattened real vectors R^n -> R^m through ``mv`` / ``rmv``
(forward / adjoint). Operators optionally carry structure that solvers
exploit:

* ``matrix``  - explicit dense ndarray or scipy.sparse matrix (row access for
  Kaczmarz, exact SVD for small problems, row/column sums for SIRT);
* ``fourier`` - (H, shape): A = F^-1 diag(H) F with the unitary 2-D/1-D DFT
  (periodic convolution), giving closed-form Tikhonov / Wiener filters and the
  Fourier "SVD" for Picard plots;
* ``nonneg``  - A maps nonnegative vectors to nonnegative vectors (SIRT,
  Richardson-Lucy, Poisson data).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LinOp:
    def __init__(self, n, m, mv, rmv, *, matrix=None, fourier=None, nonneg=False, name=""):
        self.n, self.m = int(n), int(m)
        self._mv, self._rmv = mv, rmv
        self.matrix = matrix
        self.fourier = fourier
        self.nonneg = nonneg
        self.name = name
        self._norm = None
        self._svd = None

    # -- application ---------------------------------------------------------
    def __call__(self, x):
        return self._mv(np.asarray(x, float).ravel())

    def T(self, y):
        return self._rmv(np.asarray(y, float).ravel())

    def normal(self, x):
        return self.T(self(x))

    # -- algebra ---------------------------------------------------------------
    def scaled(self, c):
        c = float(c)
        mat = None if self.matrix is None else self.matrix * c
        four = None if self.fourier is None else (self.fourier[0] * c, self.fourier[1])
        op = LinOp(self.n, self.m, lambda x: c * self._mv(x), lambda y: c * self._rmv(y),
                   matrix=mat, fourier=four, nonneg=self.nonneg and c > 0, name=self.name)
        return op

    def compose(self, other: "LinOp"):
        """self @ other."""
        mat = None
        if self.matrix is not None and other.matrix is not None:
            mat = self.matrix @ other.matrix
        return LinOp(other.n, self.m, lambda x: self._mv(other._mv(x)),
                     lambda y: other._rmv(self._rmv(y)), matrix=mat,
                     nonneg=self.nonneg and other.nonneg, name=f"{self.name}*{other.name}")

    # -- spectral info ----------------------------------------------------------
    def norm(self, n_iter=60, seed=0):
        """Spectral norm ||A||_2 (exact for small explicit matrices, power iteration otherwise)."""
        if self._norm is None:
            if self.fourier is not None:
                self._norm = float(np.abs(self.fourier[0]).max())
            elif self.matrix is not None and not sp.issparse(self.matrix) and min(self.n, self.m) <= 1500:
                self._norm = float(np.linalg.norm(self.matrix, 2))
            else:
                rng = np.random.default_rng(seed)
                v = rng.normal(size=self.n)
                v /= np.linalg.norm(v)
                s = 1.0
                for _ in range(n_iter):
                    w = self.normal(v)
                    s = float(np.linalg.norm(w))
                    if s == 0:
                        break
                    v = w / s
                self._norm = float(np.sqrt(max(s, 1e-300)))
        return self._norm

    def dense(self):
        if self.matrix is not None:
            return self.matrix.toarray() if sp.issparse(self.matrix) else np.asarray(self.matrix)
        if self.n > 4096:
            raise ValueError("operator too large to materialize")
        return np.column_stack([self(e) for e in np.eye(self.n)])

    @property
    def has_svd(self):
        return self.fourier is None and max(self.n, self.m) <= 2048 and (self.matrix is not None or self.n <= 1024)

    def svd(self):
        """Thin SVD (U, s, Vt) of the explicit matrix; cached."""
        if self._svd is None:
            U, s, Vt = np.linalg.svd(self.dense(), full_matrices=False)
            self._svd = (U, s, Vt)
        return self._svd

    def row_sums(self):
        return self(np.ones(self.n))

    def col_sums(self):
        return self.T(np.ones(self.m))


# ----------------------------------------------------------------------------- builders
def identity(n):
    return LinOp(n, n, lambda x: x.copy(), lambda y: y.copy(), fourier=None, nonneg=True, name="I",
                 matrix=sp.identity(n, format="csr"))


def from_matrix(M, nonneg=None, name=""):
    if nonneg is None:
        nonneg = bool((M.min() if not sp.issparse(M) else M.data.min(initial=0)) >= 0)
    if sp.issparse(M):
        M = M.tocsr()
        Mt = M.T.tocsr()
        return LinOp(M.shape[1], M.shape[0], lambda x: M @ x, lambda y: Mt @ y, matrix=M, nonneg=nonneg, name=name)
    M = np.ascontiguousarray(M, float)
    return LinOp(M.shape[1], M.shape[0], lambda x: M @ x, lambda y: M.T @ y, matrix=M, nonneg=nonneg, name=name)


def _ortho_fft(x, shape):
    return np.fft.fftn(x.reshape(shape), norm="ortho")


def _ortho_ifft(X, shape):
    return np.fft.ifftn(X, norm="ortho").real.ravel()


def psf_to_otf(psf, shape):
    """Embed a centered PSF in an array of `shape` and return its (non-normalized) DFT, so that
    periodic convolution x -> ifft(H * fft(x)) is applied."""
    pad = np.zeros(shape)
    sl = tuple(slice(0, s) for s in psf.shape)
    pad[sl] = psf
    for ax, s in enumerate(psf.shape):
        pad = np.roll(pad, -(s // 2), axis=ax)
    return np.fft.fftn(pad)


def convolution(psf, shape, name="conv"):
    """Periodic convolution with `psf` (1-D or 2-D). Exact circulant structure (``fourier``)."""
    shape = tuple(shape)
    H = psf_to_otf(np.asarray(psf, float), shape)
    Hc = np.conj(H)

    def mv(x):
        return np.fft.ifftn(H * np.fft.fftn(x.reshape(shape))).real.ravel()

    def rmv(y):
        return np.fft.ifftn(Hc * np.fft.fftn(y.reshape(shape))).real.ravel()

    n = int(np.prod(shape))
    return LinOp(n, n, mv, rmv, fourier=(H, shape), nonneg=bool(np.min(psf) >= 0), name=name)


def mask_op(mask, name="mask"):
    """Restriction to the pixels where `mask` is True (output = observed entries only)."""
    idx = np.flatnonzero(np.asarray(mask).ravel())
    n = mask.size

    def mv(x):
        return x[idx]

    def rmv(y):
        out = np.zeros(n)
        out[idx] = y
        return out

    M = sp.csr_matrix((np.ones(idx.size), (np.arange(idx.size), idx)), shape=(idx.size, n))
    return LinOp(n, idx.size, mv, rmv, matrix=M, nonneg=True, name=name)


def downsample(shape, factor):
    """Block averaging by `factor` in every axis (adjoint spreads uniformly)."""
    shape = tuple(shape)
    f = int(factor)
    small = tuple(s // f for s in shape)
    n, m = int(np.prod(shape)), int(np.prod(small))
    c = 1.0 / f ** len(shape)

    if len(shape) == 2:
        def mv(x):
            return x.reshape(small[0], f, small[1], f).sum(axis=(1, 3)).ravel() * c

        def rmv(y):
            return np.repeat(np.repeat(y.reshape(small), f, 0), f, 1).ravel() * c
    else:
        def mv(x):
            return x.reshape(small[0], f).sum(axis=1) * c

        def rmv(y):
            return np.repeat(y, f) * c
    return LinOp(n, m, mv, rmv, nonneg=True, name="down")


def fourier_sampling(mask, name="F_mask"):
    """MRI-style undersampled unitary DFT of a real image: y = [Re; Im](P F x).
    Adjoint of the real-stacked map is exact (x real)."""
    shape = mask.shape
    idx = np.flatnonzero(mask.ravel())
    k = idx.size
    n = int(np.prod(shape))

    def mv(x):
        X = _ortho_fft(x, shape).ravel()[idx]
        return np.concatenate([X.real, X.imag])

    def rmv(y):
        Z = np.zeros(n, complex)
        Z[idx] = y[:k] + 1j * y[k:]
        return _ortho_ifft(Z.reshape(shape), shape)

    return LinOp(n, 2 * k, mv, rmv, name=name)


def radon_matrix(N, angles, n_det=None):
    """Pixel-driven parallel-beam projector with linear interpolation onto detector bins.
    Returns a sparse (n_angles*n_det, N*N) matrix; unit pixel size, detector spacing 1."""
    n_det = n_det or int(np.ceil(np.sqrt(2) * N)) | 1
    c = (N - 1) / 2.0
    yy, xx = np.mgrid[0:N, 0:N]
    X = (xx - c).ravel()
    Y = (c - yy).ravel()
    pix = np.arange(N * N)
    rows, cols, vals = [], [], []
    d0 = (n_det - 1) / 2.0
    for i, th in enumerate(angles):
        s = X * np.cos(th) + Y * np.sin(th) + d0
        j = np.floor(s).astype(int)
        w = s - j
        for jj, ww in ((j, 1 - w), (j + 1, w)):
            ok = (jj >= 0) & (jj < n_det) & (ww > 0)
            rows.append(i * n_det + jj[ok])
            cols.append(pix[ok])
            vals.append(ww[ok])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(angles) * n_det, N * N))
    return A, n_det


def gaussian_psf(sigma, radius=None, ndim=2):
    radius = int(radius or np.ceil(3.5 * sigma))
    t = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    if ndim == 1:
        return g / g.sum()
    k = np.outer(g, g)
    return k / k.sum()


def motion_psf(length, angle_deg, size=None):
    size = size or int(length) | 1
    size = max(size, 3) | 1
    k = np.zeros((size, size))
    c = size // 2
    th = np.deg2rad(angle_deg)
    for t in np.linspace(-length / 2, length / 2, 8 * size):
        x, y = c + t * np.cos(th), c - t * np.sin(th)
        i0, j0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - i0, x - j0
        for di, dj, w in ((0, 0, (1 - fy) * (1 - fx)), (1, 0, fy * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 1, fy * fx)):
            if 0 <= i0 + di < size and 0 <= j0 + dj < size:
                k[i0 + di, j0 + dj] += w
    return k / k.sum()


def defocus_psf(radius, size=None):
    size = size or (2 * int(np.ceil(radius)) + 3)
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size] - c
    # supersampled disk
    ss = 4
    o = (np.arange(ss) + 0.5) / ss - 0.5
    k = np.zeros((size, size))
    for a in o:
        for b in o:
            k += ((yy + a) ** 2 + (xx + b) ** 2) <= radius ** 2
    return k / k.sum()


# ----------------------------------------------------------------------------- finite differences
class Grad:
    """Forward differences with Neumann boundary (last difference = 0), Chambolle convention.
    Acts on arrays of `shape` (1-D or 2-D); output has shape (ndim,) + shape."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.n = int(np.prod(self.shape))
        self.m = self.ndim * self.n

    def __call__(self, x):
        u = x.reshape(self.shape)
        out = np.zeros((self.ndim,) + self.shape)
        if self.ndim == 1:
            out[0, :-1] = u[1:] - u[:-1]
        else:
            out[0, :, :-1] = u[:, 1:] - u[:, :-1]   # d/dx (columns)
            out[1, :-1, :] = u[1:, :] - u[:-1, :]   # d/dy (rows)
        return out

    def T(self, p):
        """Adjoint = -div."""
        p = p.reshape((self.ndim,) + self.shape)
        if self.ndim == 1:
            q = p[0]
            out = np.zeros(self.shape)
            out[1:] += q[:-1]
            out[:-1] -= q[:-1]
            return out.ravel()
        px, py = p
        out = np.zeros(self.shape)
        out[:, 1:] += px[:, :-1]
        out[:, :-1] -= px[:, :-1]
        out[1:, :] += py[:-1, :]
        out[:-1, :] -= py[:-1, :]
        return out.ravel()

    def norm_sq(self):
        return 4.0 * self.ndim

    def as_linop(self):
        return LinOp(self.n, self.m, lambda x: self(x).ravel(), lambda y: self.T(y), name="D")


class SymGrad:
    """Symmetrized gradient of a vector field w = (w_x[, w_y]) built from the same Neumann forward
    differences. Channels (2-D): (dx wx, dy wy, (dy wx + dx wy)/sqrt 2), so the plain Euclidean
    norm over channels equals the Frobenius norm of the symmetric tensor. 1-D: (dw)."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.nc = 1 if self.ndim == 1 else 3

    @staticmethod
    def _d(u, axis):
        out = np.zeros_like(u)
        if axis == 0:
            out[:-1] = u[1:] - u[:-1]
        else:
            out[:, :-1] = u[:, 1:] - u[:, :-1]
        return out

    @staticmethod
    def _dT(p, axis):
        out = np.zeros_like(p)
        if axis == 0:
            out[1:] += p[:-1]
            out[:-1] -= p[:-1]
        else:
            out[:, 1:] += p[:, :-1]
            out[:, :-1] -= p[:, :-1]
        return out

    def __call__(self, w):
        w = w.reshape((self.ndim,) + self.shape)
        if self.ndim == 1:
            return self._d(w[0], 0)[None]
        wx, wy = w
        r = 1.0 / np.sqrt(2.0)
        return np.stack([self._d(wx, 1), self._d(wy, 0), r * (self._d(wx, 0) + self._d(wy, 1))])

    def T(self, q):
        q = q.reshape((self.nc,) + self.shape)
        if self.ndim == 1:
            return self._dT(q[0], 0)[None]
        qxx, qyy, qxy = q
        r = 1.0 / np.sqrt(2.0)
        return np.stack([self._dT(qxx, 1) + r * self._dT(qxy, 0), self._dT(qyy, 0) + r * self._dT(qxy, 1)])

    def norm_sq(self):
        return 4.0 if self.ndim == 1 else 8.0
