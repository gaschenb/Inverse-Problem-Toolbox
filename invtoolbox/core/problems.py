"""Inverse problems: forward model, ground truth, noise simulation.

Linear problems carry a ``LinOp`` normalized to ||A||_2 = 1, so step sizes and regularization
weights are comparable across problems. Nonlinear problems expose ``F`` and a dense Jacobian.

Noise level semantics (one slider for every model):
  gaussian       ||e|| ~= level * ||A x||                      (sigma = level ||Ax|| / sqrt m)
  poisson        peak photon count = 1 / level^2,  b = Poisson(c A x) / c
  salt-pepper    a fraction `level` of the data is replaced by min / max of A x
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from . import operators as ops
from .phantoms import IMAGES, SIGNALS, make_image, make_signal


@dataclass
class Data:
    b: np.ndarray
    kind: str
    level: float
    sigma: float          # per-entry std of the noise (Gaussian equivalent)
    delta: float          # expected ||e||
    scale: float = 1.0    # Poisson counts per unit
    seed: int = 0
    clean: np.ndarray | None = None


@dataclass
class Problem:
    key: str
    title: str
    category: str
    shape: tuple
    x_true: np.ndarray
    A: ops.LinOp | None = None
    F: callable = None
    J: callable = None
    data_kind: str = "signal"
    data_shape: tuple = ()
    equations: list = field(default_factory=list)
    note: str = ""
    noise_kinds: tuple = ("gaussian",)
    noise_default: tuple = ("gaussian", 0.01)
    nonneg: bool = True
    sign_ambiguous: bool = False
    meta: dict = field(default_factory=dict)
    x_label: str = "x"

    # -------------------------------------------------------------------- basics
    @property
    def n(self):
        return self.x_true.size

    @property
    def m(self):
        return self.A.m if self.A is not None else self.F(self.x_true).size

    @property
    def linear(self):
        return self.A is not None

    @property
    def has_truth(self):
        """False for measured data (the truth is stored as NaN)."""
        return bool(np.all(np.isfinite(self.x_true)))

    @property
    def is_image(self):
        return len(self.shape) == 2

    def forward(self, x):
        return self.A(x) if self.linear else self.F(np.asarray(x, float).ravel())

    def jacobian(self, x):
        return self.A.dense() if self.linear else self.J(np.asarray(x, float).ravel())

    def grad(self, x, b):
        """Gradient of 0.5 ||F(x) - b||^2."""
        if self.linear:
            return self.A.T(self.A(x) - b)
        return self.J(x).T @ (self.F(x) - b)

    def x0(self, b=None):
        return np.asarray(self.meta.get("x0", np.zeros(self.n)), float).copy()

    def align(self, x):
        """Resolve the global-sign ambiguity of phase retrieval against the truth."""
        if self.sign_ambiguous and np.linalg.norm(x + self.x_true) < np.linalg.norm(x - self.x_true):
            return -x
        return x

    # -------------------------------------------------------------------- noise
    def simulate(self, kind=None, level=None, seed=0) -> Data:
        if "measured" in self.meta:          # user data: the data are fixed, nothing to simulate
            return self.meta["measured"]
        kind = kind or self.noise_default[0]
        level = self.noise_default[1] if level is None else float(level)
        rng = np.random.default_rng(seed)
        y = self.forward(self.x_true)
        m = y.size
        if kind == "gaussian":
            sigma = level * np.linalg.norm(y) / np.sqrt(m)
            b = y + sigma * rng.normal(size=m)
            return Data(b, kind, level, sigma, sigma * np.sqrt(m), seed=seed, clean=y)
        if kind == "poisson":
            peak = 1.0 / max(level, 1e-4) ** 2
            c = peak / max(y.max(), 1e-12)
            b = rng.poisson(np.clip(c * y, 0, None)) / c
            delta = float(np.sqrt(np.clip(y, 0, None).sum() / c))
            return Data(b, kind, level, delta / np.sqrt(m), delta, scale=c, seed=seed, clean=y)
        if kind == "salt-pepper":
            b = y.copy()
            hit = rng.random(m) < level
            b[hit] = np.where(rng.random(hit.sum()) < .5, y.min(), y.max())
            delta = float(np.linalg.norm(b - y))
            return Data(b, kind, level, delta / np.sqrt(m), delta, seed=seed, clean=y)
        raise ValueError(f"unknown noise model {kind!r}")

    # -------------------------------------------------------------------- display helpers
    def data_image(self, b):
        """Data reshaped for display: an image (NaN where unobserved) or a 1-D array."""
        k = self.data_kind
        if k == "masked":
            img = np.full(self.n, np.nan)
            img[self.meta["idx"]] = b
            return img.reshape(self.shape)
        if k == "kspace":
            mask = self.meta["mask"]
            kk = b.size // 2
            Z = np.zeros(mask.size, complex)
            Z[np.flatnonzero(mask.ravel())] = b[:kk] + 1j * b[kk:]
            Z = np.fft.fftshift(Z.reshape(mask.shape))
            out = np.log10(1e-6 + np.abs(Z))
            out[~np.fft.fftshift(mask)] = np.nan
            return out
        if k in ("image", "sinogram", "lowres"):
            return np.asarray(b).reshape(self.data_shape)
        return np.asarray(b)


# ----------------------------------------------------------------------------- registry
@dataclass
class Spec:
    key: str
    title: str
    category: str
    builder: callable
    sizes: tuple
    size: int
    truths: tuple
    truth: str
    blurb: str


REGISTRY: dict[str, Spec] = {}
_CACHE: dict = {}


def register(key, title, category, sizes, size, truths, truth, blurb=""):
    def deco(fn):
        REGISTRY[key] = Spec(key, title, category, fn, tuple(sizes), size, tuple(truths), truth, blurb)
        return fn
    return deco


def build(key, size=None, truth=None) -> Problem:
    spec = REGISTRY[key]
    size = size or spec.size
    truth = truth or spec.truth
    ck = (key, size, truth)
    if ck not in _CACHE:
        p = spec.builder(int(size), truth)
        if p.A is not None:
            s = p.A.norm()
            p.A = p.A.scaled(1.0 / s)
            p.A._norm = 1.0
            p.meta["A_scale"] = float(s)      # physical operator = A_scale * A (used for measured data)
        _CACHE[ck] = p
    return _CACHE[ck]


def categories():
    out: dict[str, list[str]] = {}
    for k, s in REGISTRY.items():
        out.setdefault(s.category, []).append(k)
    return out


_IMG = tuple(IMAGES)
_SIG = tuple(SIGNALS)
_NOISE_IMG = ("gaussian", "poisson", "salt-pepper")


def _img_problem(key, title, N, truth, A, *, data_kind="image", data_shape=None, eq, note, noise, meta=None,
                 noise_kinds=_NOISE_IMG):
    x = make_image(truth, N).ravel()
    if not A.nonneg:
        noise_kinds = tuple(k for k in noise_kinds if k != "poisson")
    return Problem(key, title, REGISTRY[key].category, (N, N), x, A=A, data_kind=data_kind,
                   data_shape=data_shape or (N, N), equations=eq, note=note, noise_kinds=noise_kinds,
                   noise_default=noise, meta=meta or {})


# ----------------------------------------------------------------------------- imaging
@register("deblur-gaussian", "Gaussian deblurring", "Imaging", (32, 64, 96, 128), 64, _IMG, "shepp-logan",
          "Periodic convolution with a Gaussian PSF.")
def _deblur_gauss(N, truth):
    s = 2.0 * N / 64
    A = ops.convolution(ops.gaussian_psf(s), (N, N), "blur")
    return _img_problem("deblur-gaussian", "Gaussian deblurring", N, truth, A,
                        eq=[r"b = k \ast x + \varepsilon", r"\hat{k}(\omega) = e^{-\sigma^2|\omega|^2/2},\ \ \sigma = %.1f\ \mathrm{px}" % s],
                        note="The Gaussian symbol decays like exp(−σ²ω²/2): high frequencies are lost below "
                             "the noise floor and naive inversion amplifies noise without bound.",
                        noise=("gaussian", 0.01))


@register("deblur-motion", "Motion deblurring", "Imaging", (32, 64, 96, 128), 64, _IMG, "text",
          "Linear camera shake, 11 px at 30 degrees.")
def _deblur_motion(N, truth):
    L = 11 * N / 64
    A = ops.convolution(ops.motion_psf(L, 30), (N, N), "motion")
    return _img_problem("deblur-motion", "Motion deblurring", N, truth, A,
                        eq=[r"b = k_{\ell,\theta} \ast x + \varepsilon",
                            r"\hat{k}(\omega) = \mathrm{sinc}(\ell\,\omega\cdot\mathbf{e}_\theta / 2)"],
                        note="The line PSF has a sinc symbol with exact zeros along stripes in frequency space: "
                             "those directions are invisible, not merely damped.",
                        noise=("gaussian", 0.01))


@register("deblur-defocus", "Defocus deblurring", "Imaging", (32, 64, 96, 128), 64, _IMG, "bars",
          "Out-of-focus disk PSF, radius 3 px.")
def _deblur_defocus(N, truth):
    r = 3.0 * N / 64
    A = ops.convolution(ops.defocus_psf(r), (N, N), "defocus")
    return _img_problem("deblur-defocus", "Defocus deblurring", N, truth, A,
                        eq=[r"b = \frac{1}{\pi r^2}\,\mathbf{1}_{|y|\leq r} \ast x + \varepsilon",
                            r"\hat{k}(\omega) = \frac{2 J_1(r|\omega|)}{r|\omega|}"],
                        note="The Airy-type symbol 2J₁(rω)/(rω) has rings of zeros; bar patterns near those "
                             "frequencies are erased (contrast reversal between them).",
                        noise=("gaussian", 0.01))


@register("deblur-poisson", "Low-light deblurring (Poisson)", "Imaging", (32, 64, 96, 128), 64, _IMG, "stars",
          "Photon-limited astronomy-style imaging.")
def _deblur_poisson(N, truth):
    s = 1.5 * N / 64
    A = ops.convolution(ops.gaussian_psf(s), (N, N), "blur")
    return _img_problem("deblur-poisson", "Low-light deblurring (Poisson)", N, truth, A,
                        eq=[r"b \sim \mathrm{Poisson}(k \ast x),\qquad x \geq 0",
                            r"-\log p(b\,|\,x) = \sum_i\, (Ax)_i - b_i \log (Ax)_i"],
                        note="Noise variance equals the signal: least squares over-weights bright pixels. The "
                             "Kullback-Leibler data term (Richardson-Lucy, PDHG-KL) is the matched likelihood.",
                        noise=("poisson", 0.1))


@register("denoise", "Denoising", "Imaging", (32, 64, 96, 128), 64, _IMG, "blocks", "Identity forward operator.")
def _denoise(N, truth):
    A = ops.LinOp(N * N, N * N, lambda x: x.copy(), lambda y: y.copy(), fourier=(np.ones((N, N)), (N, N)),
                  nonneg=True, name="I")
    return _img_problem("denoise", "Denoising", N, truth, A,
                        eq=[r"b = x + \varepsilon"],
                        note="A = I is perfectly conditioned, yet the maximum-likelihood estimate is the noisy "
                             "image itself; the prior does all the work. Try salt-and-pepper noise with an L1 data term.",
                        noise=("gaussian", 0.15))


@register("inpainting", "Inpainting", "Imaging", (32, 64, 96, 128), 64, _IMG, "blocks",
          "70 % of pixels missing plus scratches.")
def _inpaint(N, truth):
    rng = np.random.default_rng(4)
    keep = rng.random((N, N)) > 0.7
    for r in rng.integers(0, N, 3):
        keep[r, :] = False
    keep[:, N // 3] = False
    A = ops.mask_op(keep, "P_Omega")
    p = _img_problem("inpainting", "Inpainting", N, truth, A, data_kind="masked", data_shape=(int(keep.sum()),),
                     eq=[r"b = P_\Omega\, x + \varepsilon", r"|\Omega| / N^2 = %.0f\%%" % (100 * keep.mean())],
                     note="The null space is every image supported on the missing pixels: data say nothing there, "
                          "the regularizer fills it in (harmonic for Tikhonov-gradient, piecewise-flat for TV).",
                     noise=("gaussian", 0.02), meta={"mask": keep, "idx": np.flatnonzero(keep)})
    return p


@register("super-resolution", "Super-resolution (x2)", "Imaging", (32, 64, 96, 128), 64, _IMG, "text",
          "Blur then 2x2 block-average decimation.")
def _superres(N, truth):
    s = 0.8 * N / 64
    A = ops.downsample((N, N), 2).compose(ops.convolution(ops.gaussian_psf(s), (N, N)))
    A.nonneg = True
    return _img_problem("super-resolution", "Super-resolution (x2)", N, truth, A, data_kind="lowres",
                        data_shape=(N // 2, N // 2),
                        eq=[r"b = S_2\,(k \ast x) + \varepsilon,\qquad S_2: \mathbb{R}^{N\times N}\to\mathbb{R}^{N/2\times N/2}"],
                        note="Four unknowns per measurement: a 3N²/4-dimensional null space (aliasing patterns) "
                             "on top of the blur.",
                        noise=("gaussian", 0.01))


def _ct(key, title, N, truth, angles, noise, note):
    M, nd = ops.radon_matrix(N, angles)
    A = ops.from_matrix(M, nonneg=True, name="Radon")
    deg = np.rad2deg(angles)
    return _img_problem(key, title, N, truth, A, data_kind="sinogram", data_shape=(len(angles), nd),
                        eq=[r"b = \mathcal{R}x + \varepsilon",
                            r"(\mathcal{R}x)(\theta, s) = \int_{\mathbb{R}} x(s\,\theta + t\,\theta^{\perp})\,dt",
                            r"%d\ \mathrm{angles\ in}\ [%.0f^\circ, %.0f^\circ)" % (len(angles), deg[0], deg[-1] + deg[1] - deg[0])],
                        note=note, noise=noise, meta={"angles": angles, "n_det": nd})


@register("ct-sparse-view", "CT, sparse view", "Tomography", (32, 64, 96, 128), 64, _IMG, "shepp-logan",
          "Parallel-beam CT with 30 projections.")
def _ct_sparse(N, truth):
    ang = np.linspace(0, np.pi, max(8, int(30 * N / 64)), endpoint=False)
    return _ct("ct-sparse-view", "CT, sparse view", N, truth, ang, ("gaussian", 0.02),
               "Fewer projections than the angular Nyquist rate: filtered back-projection streaks; the system "
               "is underdetermined and TV / nonnegativity close the gap.")


@register("ct-limited-angle", "CT, limited angle", "Tomography", (32, 64, 96, 128), 64, _IMG, "shepp-logan",
          "120-degree wedge of projections.")
def _ct_limited(N, truth):
    ang = np.linspace(0, np.deg2rad(120), int(80 * N / 64), endpoint=False)
    return _ct("ct-limited-angle", "CT, limited angle", N, truth, ang, ("gaussian", 0.01),
               "A missing wedge of Fourier space (Fourier slice theorem): edges tangent to the missing "
               "directions are invisible, a severely ill-posed microlocal null space.")


def _mri_mask_cartesian(N, frac=0.3, center=0.08, seed=2):
    rng = np.random.default_rng(seed)
    k = np.abs(np.arange(N) - N // 2) / (N / 2)
    p = (1 - k) ** 3
    nc = max(2, int(center * N))
    target = int(frac * N)
    p[N // 2 - nc // 2:N // 2 + nc // 2 + 1] = 0
    p = p / p.sum() * max(target - nc, 1)
    p = np.clip(p, 0, 1)
    rows = rng.random(N) < p
    rows[N // 2 - nc // 2:N // 2 + nc // 2 + 1] = True
    mask = np.zeros((N, N), bool)
    mask[rows, :] = True
    return np.fft.ifftshift(mask)


def _mri_mask_radial(N, spokes=24):
    mask = np.zeros((N, N), bool)
    c = N // 2
    r = np.linspace(-N / 2, N / 2, 4 * N)
    for th in np.linspace(0, np.pi, spokes, endpoint=False):
        i = np.clip(np.round(c + r * np.sin(th)).astype(int), 0, N - 1)
        j = np.clip(np.round(c + r * np.cos(th)).astype(int), 0, N - 1)
        mask[i, j] = True
    return np.fft.ifftshift(mask)


def _mri(key, title, N, truth, mask, descr):
    A = ops.fourier_sampling(mask)
    return _img_problem(key, title, N, truth, A, data_kind="kspace", data_shape=(N, N),
                        eq=[r"b = P_\Omega\,\mathcal{F}x + \varepsilon,\qquad \mathcal{F}^*\mathcal{F} = I",
                            descr + r",\ \ |\Omega|/N^2 = %.0f\%%" % (100 * mask.mean())],
                        note="Zero-filling (Aᵀb) gives coherent aliasing; with incoherent sampling, sparsity in "
                             "TV or wavelets makes the problem compressed-sensing recoverable.",
                        noise=("gaussian", 0.02), meta={"mask": mask}, noise_kinds=("gaussian",))


@register("mri-cartesian", "MRI, Cartesian undersampling", "Tomography", (32, 64, 96, 128), 64, _IMG,
          "shepp-logan", "Variable-density random phase-encode lines.")
def _mri_cart(N, truth):
    return _mri("mri-cartesian", "MRI, Cartesian undersampling", N, truth, _mri_mask_cartesian(N),
                r"\mathrm{variable\ density\ phase\ encodes}")


@register("mri-radial", "MRI, radial spokes", "Tomography", (32, 64, 96, 128), 64, _IMG, "shepp-logan",
          "24 radial spokes on the Cartesian grid.")
def _mri_rad(N, truth):
    return _mri("mri-radial", "MRI, radial spokes", N, truth, _mri_mask_radial(N, max(8, int(24 * N / 64))),
                r"\mathrm{radial\ spokes}")


# ----------------------------------------------------------------------------- 1-D
def _sig_problem(key, title, n, truth, A, eq, note, noise, data_kind="signal", meta=None, x_label="t"):
    x = make_signal(truth, n)
    kinds = ("gaussian", "poisson", "salt-pepper") if (A.nonneg and x.min() >= 0) else ("gaussian", "salt-pepper")
    return Problem(key, title, REGISTRY[key].category, (n,), x, A=A, data_kind=data_kind, data_shape=(A.m,),
                   equations=eq, note=note, noise_kinds=kinds, noise_default=noise, nonneg=bool(x.min() >= 0),
                   meta=meta or {}, x_label=x_label)


@register("compressed-sensing", "Compressed sensing", "Signals", (128, 256, 512), 256, _SIG, "sparse",
          "Random Gaussian matrix, m = n / 3.")
def _cs(n, truth):
    m = n * 5 // 16
    rng = np.random.default_rng(7)
    A = ops.from_matrix(rng.normal(0, 1 / np.sqrt(m), (m, n)), nonneg=False, name="Gaussian")
    return _sig_problem("compressed-sensing", "Compressed sensing", n, truth, A,
                        [r"b = Ax + \varepsilon,\qquad A_{ij} \sim \mathcal{N}(0, 1/m),\ \ m = %d \ll n = %d" % (m, n)],
                        "m << n: an (n - m)-dimensional null space. For s-sparse x and m ~ s log(n/s), l1 recovers x "
                        "exactly (restricted isometry); l2 spreads energy over every coordinate.",
                        ("gaussian", 0.02), x_label="index")


@register("spike-deconvolution", "Spike deconvolution", "Signals", (128, 256, 512), 256, _SIG, "spikes",
          "Seismic-style blurred spike train.")
def _spikes(n, truth):
    A = ops.convolution(ops.gaussian_psf(2.0 * n / 256, ndim=1), (n,), "blur")
    A.nonneg = True
    return _sig_problem("spike-deconvolution", "Spike deconvolution", n, truth, A,
                        [r"b = g_\sigma \ast x + \varepsilon,\qquad x = \sum_k a_k\,\delta_{t_k}"],
                        "Neighbouring spikes merge below the Rayleigh limit; least squares smears them, ℓ¹ and "
                        "OMP separate them when the minimum separation exceeds ~2σ.",
                        ("gaussian", 0.02), x_label="t")


def _quad(n, a, b):
    h = (b - a) / n
    return a + (np.arange(n) + 0.5) * h, h


@register("gravity", "Gravity surveying", "Integral equations", (64, 128, 256), 128, _SIG, "smooth",
          "Hansen's gravity test problem, depth d = 0.25.")
def _gravity(n, truth):
    t, h = _quad(n, 0, 1)
    d = 0.25
    K = h * d / (d ** 2 + (t[:, None] - t[None, :]) ** 2) ** 1.5
    return _sig_problem("gravity", "Gravity surveying", n, truth, ops.from_matrix(K, name="gravity"),
                        [r"b(s) = \int_0^1 \frac{d}{\left(d^2 + (s - t)^2\right)^{3/2}}\,x(t)\,dt,\qquad d = 0.25"],
                        "First-kind Fredholm equation with a smooth kernel: singular values decay geometrically, "
                        "so the Picard condition fails beyond the first ~20 modes at 1 % noise.",
                        ("gaussian", 0.01))


@register("shaw", "Shaw (1-D image restoration)", "Integral equations", (64, 128, 256), 128, _SIG, "smooth",
          "Hansen's shaw test problem.")
def _shaw(n, truth):
    t, h = _quad(n, -np.pi / 2, np.pi / 2)
    S, T = np.meshgrid(t, t, indexing="ij")
    u = np.pi * (np.sin(S) + np.sin(T))
    sinc = np.where(np.abs(u) < 1e-12, 1.0, np.sin(u) / np.where(np.abs(u) < 1e-12, 1, u))
    K = h * (np.cos(S) + np.cos(T)) ** 2 * sinc ** 2
    return _sig_problem("shaw", "Shaw (1-D image restoration)", n, truth, ops.from_matrix(K, name="shaw"),
                        [r"b(s) = \int_{-\pi/2}^{\pi/2} (\cos s + \cos t)^2 \left(\frac{\sin u}{u}\right)^2 x(t)\,dt",
                         r"u = \pi\,(\sin s + \sin t)"],
                        "Severely ill-posed: σₖ decays faster than exponentially, cond(A) ≈ 10¹⁸ at n = 128.",
                        ("gaussian", 0.01))


@register("backward-heat", "Backward heat equation", "Integral equations", (64, 128, 256), 128, _SIG,
          "piecewise", "Recover the initial temperature from u(T).")
def _heat(n, truth):
    t = (np.arange(1, n + 1)) / (n + 1)
    k = np.arange(1, n + 1)
    V = np.sqrt(2 / (n + 1)) * np.sin(np.pi * np.outer(t, k))
    T = 0.0008
    K = V @ np.diag(np.exp(-(k * np.pi) ** 2 * T)) @ V.T
    return _sig_problem("backward-heat", "Backward heat equation", n, truth, ops.from_matrix(K, nonneg=True, name="heat"),
                        [r"\partial_t u = \partial_{ss} u,\quad u(0,\cdot) = x,\quad u(t, 0) = u(t, 1) = 0",
                         r"b = u(T, \cdot) + \varepsilon = \sum_k e^{-k^2\pi^2 T}\,\langle x, \phi_k\rangle\,\phi_k + \varepsilon"],
                        "Exponentially severe: mode k is damped by exp(−k²π²T). Jumps in x0 cannot be recovered "
                        "beyond the first ~15 sine modes.",
                        ("gaussian", 0.005))


@register("differentiation", "Numerical differentiation", "Integral equations", (64, 128, 256), 128, _SIG,
          "piecewise", "Invert the Volterra integration operator.")
def _deriv(n, truth):
    t, h = _quad(n, 0, 1)
    K = np.tril(np.ones((n, n))) * h
    K[np.diag_indices(n)] = h / 2
    return _sig_problem("differentiation", "Numerical differentiation", n, truth, ops.from_matrix(K, name="int"),
                        [r"b(s) = \int_0^s x(t)\,dt + \varepsilon,\qquad x = b'"],
                        "Mildly ill-posed (σₖ ∼ 1/k): differentiation amplifies noise linearly in frequency. "
                        "TV gives the piecewise-constant derivative of a piecewise-linear signal.",
                        ("gaussian", 0.02))


@register("inverse-laplace", "Inverse Laplace transform", "Integral equations", (64, 128, 256), 128, _SIG,
          "smooth", "Relaxation-spectrum style inversion.")
def _laplace(n, truth):
    t, h = _quad(n, 0, 1)
    s = np.linspace(0, 1, n) * 10 + 0.1
    K = h * np.exp(-np.outer(s, t * 5))
    return _sig_problem("inverse-laplace", "Inverse Laplace transform", n, truth, ops.from_matrix(K, name="laplace"),
                        [r"b(s) = \int_0^{\infty} e^{-s\tau}\,x(\tau)\,d\tau,\qquad \tau = 5t,\ \ s \in [0.1, 10.1]"],
                        "Exponentially ill-posed with a positive kernel: only a handful of singular components "
                        "survive, so recovered spectra are inevitably smooth.",
                        ("gaussian", 0.005))


# ----------------------------------------------------------------------------- nonlinear
@register("phase-retrieval", "Phase retrieval", "Nonlinear", (32, 64, 128), 64, _SIG, "two-bumps",
          "Real phaseless Gaussian measurements, m = 6n.")
def _phase(n, truth):
    m = 6 * n
    rng = np.random.default_rng(3)
    A = rng.normal(0, 1 / np.sqrt(n), (m, n))
    x = make_signal(truth, n)

    def F(v):
        return (A @ v) ** 2

    def J(v):
        return 2 * (A @ v)[:, None] * A

    return Problem("phase-retrieval", "Phase retrieval", "Nonlinear", (n,), x, F=F, J=J, data_kind="signal",
                   data_shape=(m,), equations=[r"b_i = \left(a_i^{T} x\right)^2 + \varepsilon_i,\qquad i = 1,\dots,%d" % m,
                                               r"a_i \sim \mathcal{N}(0, I/n)"],
                   note="Only intensities are measured: the loss is nonconvex and x is identifiable up to a global "
                        "sign. Spectral initialization lands in the basin of attraction for m >~ 4n.",
                   noise_kinds=("gaussian",), noise_default=("gaussian", 0.01), nonneg=False, sign_ambiguous=True,
                   meta={"A": A}, x_label="t")


@register("coefficient-id", "Diffusion coefficient identification", "Nonlinear", (50, 100, 200), 100,
          ("smooth", "piecewise", "mixed"), "smooth", "Recover log a(t) in -(a u')' = f from three experiments.")
def _coeff(n, truth):
    N = n                     # coefficient cells; nodes 1..N-1 interior
    h = 1.0 / N
    tn = np.arange(1, N) * h
    tc = (np.arange(N) + 0.5) * h
    fs = [np.ones(N - 1), np.sin(np.pi * tn) * 4, 6 * np.exp(-((tn - 0.7) / 0.1) ** 2)]
    obs = np.arange(0, N - 1, 2)
    x = 1.2 * make_signal(truth, n) - 0.4

    def stiffness(mv):
        a = np.exp(np.clip(mv, -30, 30))
        diag = (a[:-1] + a[1:]) / h ** 2
        off = -a[1:-1] / h ** 2
        ab = np.zeros((3, N - 1))
        ab[0, 1:] = off
        ab[1] = diag
        ab[2, :-1] = off
        return a, ab

    def states(mv):
        a, ab = stiffness(mv)
        return a, ab, [solve_banded((1, 1), ab, f) for f in fs]

    def F(mv):
        _, _, us = states(mv)
        return np.concatenate([u[obs] for u in us])

    def J(mv):
        a, ab, us = states(mv)
        blocks = []
        for u in us:
            ue = np.concatenate([[0.0], u, [0.0]])
            du = np.diff(ue)                       # u_{j+1} - u_j over cells j = 0..N-1
            # d(K u)/d m_j = a_j * (flux contribution) on the two nodes bounding cell j
            B = np.zeros((N - 1, N))
            j = np.arange(N)
            left, right = j - 1, j            # interior node indices (0-based) of cell j's ends
            ok = left >= 0
            B[left[ok], j[ok]] += -a[ok] * du[ok] / h ** 2
            ok = right <= N - 2
            B[right[ok], j[ok]] += a[ok] * du[ok] / h ** 2
            S = -solve_banded((1, 1), ab, B)
            blocks.append(S[obs])
        return np.vstack(blocks)

    return Problem("coefficient-id", "Diffusion coefficient identification", "Nonlinear", (n,), x, F=F, J=J,
                   data_kind="multi", data_shape=(len(fs) * obs.size,),
                   equations=[r"-\left(e^{x(t)}\,u_k'(t)\right)' = f_k(t),\quad u_k(0) = u_k(1) = 0,\quad k = 1,2,3",
                              r"b = \left(u_k(t_j)\right)_{k,j} + \varepsilon"],
                   note="The map x ↦ u is smoothing twice over (an elliptic solve, then point evaluation of a "
                        "smooth state): the linearized problem is severely ill-posed, and x is only identifiable "
                        "where uₖ′ does not vanish.",
                   noise_kinds=("gaussian",), noise_default=("gaussian", 0.01), nonneg=False,
                   meta={"x0": np.zeros(n), "obs": obs, "n_exp": len(fs), "tc": tc}, x_label="t")
