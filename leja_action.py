from __future__ import annotations
import numpy as np
import scipy.sparse as sp
# from .divided_differences import DividedDifferencesLog

class LejaLogAction:
    """
    Compute y ≈ log(Q)v using Newton–Leja interpolation with a posteriori stop:
        e_m = |d_m| * ||w_m||_2 <= tol
    """
    def __init__(self, divided_diff: DividedDifferencesLog):
        self.dd = divided_diff

    @staticmethod
    def _apply_Q(Q, y: np.ndarray) -> np.ndarray:
        if callable(Q):
            return Q(y)
        if sp.issparse(Q):
            return Q.dot(y)
        return Q @ y

    def apply(
        self,
        v: np.ndarray,
        Q,
        c: float,
        Gamma: float,
        leja_points: np.ndarray,
        alpha: float,
        beta: float,
        m_max: int,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, int, float]:
        v = np.asarray(v, dtype=float).reshape(-1)
        w = v.copy()

        # coefficients d_m
        d, _ = self.dd.compute(alpha=alpha, beta=beta, leja_points=leja_points, m=m_max)
        d = np.asarray(d, dtype=float)
        xi = np.asarray(leja_points[: m_max + 1], dtype=float)

        max_pts = min(len(d), len(xi), m_max + 1)
        d = d[:max_pts]
        xi = xi[:max_pts]

        poly = d[0] * v

        m = 1
        e_m = abs(d[m]) * np.linalg.norm(w) if m < max_pts else 0.0

        while m < max_pts and e_m > tol:
            Qw = self._apply_Q(Q, w)
            # Newton basis recurrence (your formula)
            w = (Qw - c * w) / Gamma - xi[m - 1] * w
            poly += d[m] * w

            m += 1
            if m < max_pts:
                e_m = abs(d[m]) * np.linalg.norm(w)

        return poly, m, float(e_m)