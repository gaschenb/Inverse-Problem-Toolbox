"""Spectral (SVD / Fourier) decompositions for filter-based methods, Picard plots and closed-form
parameter-choice curves.

For an explicit small matrix, A = U diag(s) V^T. For a periodic convolution, A = F^* diag(H) F with
the unitary DFT, which is an SVD with s = |H| up to phases. Both expose
    beta   = coefficients of b in the left singular basis,
    s      = singular values (same ordering),
    x(phi) = sum_i phi_i beta_i / s_i v_i       (filtered solution).
"""
from __future__ import annotations

import numpy as np


class Spectral:
    def __init__(self, problem):
        A = problem.A
        self.problem = problem
        if A.fourier is not None:
            H, shape = A.fourier
            self.kind = "fourier"
            self.shape = shape
            self.H = H.ravel()
            self.s = np.abs(self.H)
            self.m = self.n = self.H.size
        elif A.has_svd:
            self.kind = "svd"
            self.U, self.s, self.Vt = A.svd()
            self.m, self.n = A.m, A.n
        else:
            raise ValueError("no spectral decomposition for this operator")

    @staticmethod
    def available(problem):
        A = problem.A
        return A is not None and (A.fourier is not None or A.has_svd)

    def beta(self, b):
        if self.kind == "fourier":
            return np.fft.fftn(b.reshape(self.shape), norm="ortho").ravel()
        return self.U.T @ b

    def out_of_range_sq(self, b, beta):
        """||b||^2 - ||U^T b||^2 (component of b outside range(U))."""
        return max(float(b @ b) - float(np.sum(np.abs(beta) ** 2)), 0.0)

    def solve(self, b, phi, beta=None):
        beta = self.beta(b) if beta is None else beta
        if self.kind == "fourier":
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(self.s > 1e-300, phi * beta / np.where(self.s > 1e-300, self.H, 1), 0)
            return np.fft.ifftn(q.reshape(self.shape), norm="ortho").real.ravel()
        q = np.where(self.s > 1e-300, phi * beta / np.maximum(self.s, 1e-300), 0)
        return self.Vt.T @ q

    def sorted(self, b):
        """(s, |beta|) sorted by decreasing s, for Picard plots (Fourier: real-signal duplicates kept)."""
        beta = np.abs(self.beta(b))
        o = np.argsort(-self.s)
        return self.s[o], beta[o]

    # ------------------------------------------------------------------ filters
    @staticmethod
    def tikhonov(s, lam):
        s2 = s * s
        return s2 / (s2 + lam)

    @staticmethod
    def tsvd(s, k):
        phi = np.zeros_like(s)
        o = np.argsort(-s)
        phi[o[:int(k)]] = 1.0
        return phi

    def curves(self, b, lams, delta=None):
        """Closed-form Tikhonov curves: residual norm, solution norm, GCV, error vs truth."""
        beta = self.beta(b)
        ab2 = np.abs(beta) ** 2
        s2 = self.s ** 2
        oor = self.out_of_range_sq(b, beta)
        xt = self.problem.x_true
        res, sol, gcv, err = [], [], [], []
        for lam in lams:
            phi = s2 / (s2 + lam)
            r2 = float(np.sum((1 - phi) ** 2 * ab2)) + oor
            with np.errstate(divide="ignore", invalid="ignore"):
                x2 = float(np.sum(np.where(self.s > 1e-300, phi ** 2 * ab2 / np.maximum(s2, 1e-300), 0)))
            dof = self.m - float(phi.sum())
            res.append(np.sqrt(r2))
            sol.append(np.sqrt(x2))
            gcv.append(r2 / max(dof, 1e-12) ** 2)
            x = self.solve(b, phi, beta)
            err.append(np.linalg.norm(x - xt) / np.linalg.norm(xt))
        return {"lam": np.asarray(lams), "res": np.array(res), "reg": np.array(sol), "gcv": np.array(gcv),
                "err": np.array(err)}
