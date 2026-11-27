"""Orthonormal periodized discrete wavelet transforms (1-D and separable 2-D, Mallat layout).

W is orthogonal (W^T W = I) for the Haar / Daubechies filters below, so the l1-in-wavelets prox is
exact: prox(x) = W^T soft(W x). Coefficients are stored in an array of the signal's shape.
"""
from __future__ import annotations

import numpy as np

_S3 = np.sqrt(3.0)
FILTERS = {
    "haar": np.array([1.0, 1.0]) / np.sqrt(2.0),
    "db2": np.array([1 + _S3, 3 + _S3, 3 - _S3, 1 - _S3]) / (4 * np.sqrt(2.0)),
    "db4": np.array([0.2303778133088964, 0.7148465705529154, 0.6308807679298587, -0.0279837694168599,
                     -0.1870348117190931, 0.0308413818355607, 0.0328830116668852, -0.0105974017850690]),
}


def _hg(name):
    h = FILTERS[name]
    L = h.size
    g = np.array([(-1) ** k * h[L - 1 - k] for k in range(L)])
    return h, g


def _analysis_axis0(x, h, g):
    N = x.shape[0]
    half = N // 2
    a = np.zeros((half,) + x.shape[1:])
    d = np.zeros_like(a)
    base = 2 * np.arange(half)
    for k in range(h.size):
        xi = x[(base + k) % N]
        a += h[k] * xi
        d += g[k] * xi
    return np.concatenate([a, d], axis=0)


def _synthesis_axis0(c, h, g):
    N = c.shape[0]
    half = N // 2
    a, d = c[:half], c[half:]
    x = np.zeros_like(c)
    base = 2 * np.arange(half)
    for k in range(h.size):
        x[(base + k) % N] += h[k] * a + g[k] * d
    return x


def max_level(shape, name):
    L = FILTERS[name].size
    n = min(shape)
    lev = 0
    while n % 2 == 0 and n // 2 >= L:
        n //= 2
        lev += 1
    return lev


class Wavelet:
    def __init__(self, shape, name="db2", levels=None):
        self.shape = tuple(shape)
        self.name = name
        self.h, self.g = _hg(name)
        ml = max_level(self.shape, name)
        self.levels = ml if levels is None else int(min(levels, ml))

    def forward(self, x):
        c = np.array(x, float).reshape(self.shape)
        sz = list(self.shape)
        for _ in range(self.levels):
            if len(self.shape) == 1:
                c[:sz[0]] = _analysis_axis0(c[:sz[0]], self.h, self.g)
            else:
                blk = c[:sz[0], :sz[1]]
                blk = _analysis_axis0(blk, self.h, self.g)
                blk = _analysis_axis0(blk.T, self.h, self.g).T
                c[:sz[0], :sz[1]] = blk
            sz = [s // 2 for s in sz]
        return c.ravel()

    def inverse(self, c):
        x = np.array(c, float).reshape(self.shape)
        sizes = []
        sz = list(self.shape)
        for _ in range(self.levels):
            sizes.append(list(sz))
            sz = [s // 2 for s in sz]
        for sz in reversed(sizes):
            if len(self.shape) == 1:
                x[:sz[0]] = _synthesis_axis0(x[:sz[0]], self.h, self.g)
            else:
                blk = x[:sz[0], :sz[1]]
                blk = _synthesis_axis0(blk.T, self.h, self.g).T
                blk = _synthesis_axis0(blk, self.h, self.g)
                x[:sz[0], :sz[1]] = blk
        return x.ravel()

    def detail_mask(self):
        """True for detail coefficients (the coarsest approximation block is left unpenalized)."""
        m = np.ones(self.shape, bool)
        sz = [s >> self.levels for s in self.shape]
        m[tuple(slice(0, s) for s in sz)] = False
        return m.ravel()


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def ti_denoise(x, shape, thresh, name="db2", shifts=4):
    """Translation-invariant (cycle-spinning) wavelet soft-threshold denoiser."""
    W = Wavelet(shape, name)
    mask = W.detail_mask()
    img = x.reshape(shape)
    acc = np.zeros(shape)
    sh = range(shifts)
    cnt = 0
    for s1 in sh:
        for s2 in (sh if len(shape) == 2 else [0]):
            ax = (0, 1) if len(shape) == 2 else (0,)
            shift = (s1, s2) if len(shape) == 2 else (s1,)
            y = np.roll(img, shift, axis=ax)
            c = W.forward(y)
            c[mask] = soft(c[mask], thresh)
            acc += np.roll(W.inverse(c).reshape(shape), tuple(-s for s in shift), axis=ax)
            cnt += 1
    return (acc / cnt).ravel()
