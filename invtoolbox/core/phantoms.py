"""Synthetic ground truths: images on an N x N grid and signals on t in [0, 1], scaled to [0, 1]."""
from __future__ import annotations

import numpy as np

# modified Shepp-Logan (Toft): value, a, b, x0, y0, phi
_SL = [(1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0), (-.2, .11, .31, .22, 0, -18),
       (-.2, .16, .41, -.22, 0, 18), (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0),
       (.1, .046, .046, 0, -.1, 0), (.1, .046, .023, -.08, -.605, 0), (.1, .023, .023, 0, -.606, 0),
       (.1, .023, .046, .06, -.605, 0)]


def _grid(N, ss=1):
    t = (np.arange(N * ss) + 0.5) / (N * ss) * 2 - 1
    X, Y = np.meshgrid(t, -t)
    return X, Y


def _down(img, ss):
    if ss == 1:
        return img
    N = img.shape[0] // ss
    return img.reshape(N, ss, N, ss).mean(axis=(1, 3))


def shepp_logan(N):
    ss = 3
    X, Y = _grid(N, ss)
    img = np.zeros_like(X)
    for v, a, b, x0, y0, phi in _SL:
        p = np.deg2rad(phi)
        xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
        yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += v
    return _down(img, ss)


def blocks(N):
    ss = 3
    X, Y = _grid(N, ss)
    img = np.zeros_like(X)
    img[(np.abs(X + .35) < .35) & (np.abs(Y - .3) < .4)] = .6
    img[(X - .35) ** 2 + (Y - .35) ** 2 < .27 ** 2] = 1.0
    img[(np.abs(X - .25) < .5) & (np.abs(Y + .45) < .2)] = .35
    img[((X + .4) ** 2 + (Y + .4) ** 2 < .22 ** 2)] = .85
    img[(np.abs(X - .4) < .08) & (np.abs(Y - .35) < .08)] = .3
    return _down(img, ss)


def bumps(N):
    X, Y = _grid(N)
    img = (np.exp(-((X + .3) ** 2 + (Y - .25) ** 2) / .08) + .8 * np.exp(-((X - .35) ** 2 + (Y + .1) ** 2) / .03)
           + .5 * np.exp(-((X - .05) ** 2 + (Y + .55) ** 2) / .02) + .3 * np.exp(-(X ** 2 + Y ** 2) / .5))
    return img / img.max()


def text(N, s="INV"):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    ss = 3
    tp = TextPath((0, 0), s, size=1.0, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    v = tp.vertices
    lo, hi = v.min(0), v.max(0)
    scale = 1.6 / max(hi - lo)
    X, Y = _grid(N, ss)
    pts = np.column_stack([X.ravel(), Y.ravel()]) / scale + (lo + hi) / 2
    inside = tp.contains_points(pts).reshape(X.shape).astype(float)
    # TextPath with holes: contains_points uses nonzero rule on compound path; fine for block letters
    img = _down(inside, ss)
    return 0.1 + 0.9 * img


def stars(N, seed=5):
    rng = np.random.default_rng(seed)
    X, Y = _grid(N)
    img = 0.15 * np.exp(-((X + .1) ** 2 / .3 + (Y - .1) ** 2 / .1))
    k = max(12, N // 4)
    ij = rng.integers(2, N - 2, size=(k, 2))
    amp = rng.uniform(.2, 1.0, k) ** 2
    for (i, j), a in zip(ij, amp):
        img[i, j] += a
    return img / img.max()


def bars(N):
    img = np.zeros((N, N))
    widths = [max(1, int(round(N / w))) for w in (12, 16, 22, 32)]
    x0 = N // 10
    rows = [(N // 10, N // 2 - N // 20), (N // 2 + N // 20, N - N // 10)]
    for gi, w in enumerate(widths):
        r0, r1 = rows[gi // 2]
        start = x0 + (gi % 2) * (N // 2 - N // 10)
        for b in range(3):
            c = start + 2 * b * w
            img[r0:r1, c:c + w] = 1.0
    img[N // 2 - 1:N // 2 + 1, :] = 0.5
    return img


IMAGES = {"shepp-logan": shepp_logan, "blocks": blocks, "bumps": bumps, "text": text, "stars": stars, "bars": bars}


# ----------------------------------------------------------------------------- 1-D
def sig_smooth(t):
    x = 0.8 * np.exp(-((t - .3) / .1) ** 2) + np.exp(-((t - .68) / .16) ** 2)
    return x / x.max()


def sig_piecewise(t):
    x = np.zeros_like(t)
    x[(t > .1) & (t <= .25)] = .6
    x[(t > .25) & (t <= .4)] = 1.0
    x[(t > .55) & (t <= .7)] = .3
    x[(t > .7) & (t <= .85)] = .75
    return x


def sig_mixed(t):
    x = 0.5 * np.sin(2 * np.pi * t) ** 2 * (t < .5) + (t >= .55) * (t < .75) * .8 + (t >= .75) * (1 - t) * 2.5
    x += 0.4 * np.exp(-((t - .25) / .05) ** 2)
    return x / x.max()


def sig_spikes(t, k=10, seed=11):
    rng = np.random.default_rng(seed)
    n = t.size
    x = np.zeros(n)
    idx = rng.choice(np.arange(n // 20, n - n // 20), k, replace=False)
    x[idx] = rng.uniform(.4, 1.0, k) * np.where(rng.random(k) < .3, -1, 1)
    return x


def sig_sparse_signed(t, k=12, seed=7):
    rng = np.random.default_rng(seed)
    n = t.size
    x = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    x[idx] = rng.normal(0, 1, k) + np.sign(rng.normal(0, 1, k))
    return x / np.abs(x).max()


def sig_two_bumps(t):
    x = np.exp(-0.5 * ((t - 0.3) / 0.08) ** 2) + 0.7 * np.exp(-0.5 * ((t - 0.65) / 0.05) ** 2)
    return x / x.max()


SIGNALS = {"smooth": sig_smooth, "piecewise": sig_piecewise, "mixed": sig_mixed, "spikes": sig_spikes,
           "sparse": sig_sparse_signed, "two-bumps": sig_two_bumps}


def make_image(name, N):
    return np.clip(IMAGES[name](N), 0, None)


def make_signal(name, n):
    t = (np.arange(n) + 0.5) / n
    return SIGNALS[name](t)
