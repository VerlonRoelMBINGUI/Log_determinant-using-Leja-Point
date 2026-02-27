from __future__ import annotations
import numpy as np

class DividedDifferencesLog:
    """
    Compute divided differences of log(x) on [alpha, beta] using
    Leja matrix + truncated Taylor for log(I+W)e1.
    """
    def __init__(self, taylor_degree: int = 120):
        self.p = int(taylor_degree)

    @staticmethod
    def leja_matrix(xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        m = len(xi) - 1
        L = np.zeros((m + 1, m + 1), dtype=float)
        L[0, 0] = xi[0]
        for i in range(1, m + 1):
            L[i, i] = xi[i]
            L[i, i - 1] = 1.0
        return L

    @staticmethod
    def log1p_taylor_matvec(W: np.ndarray, e1: np.ndarray, p: int) -> np.ndarray:
        """
        y = log(I+W)e1 approx by sum_{k=1}^p (-1)^{k+1} W^k e1 / k
        """
        y = np.zeros_like(e1, dtype=float)
        Wk_e1 = e1.astype(float).copy()
        for k in range(1, p + 1):
            Wk_e1 = W @ Wk_e1
            y += ((-1) ** (k + 1)) * Wk_e1 / k
        return y

    def compute(self, alpha: float, beta: float, leja_points: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        alpha = float(alpha); beta = float(beta)
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for log(x).")
        if beta < alpha:
            raise ValueError("beta must be >= alpha.")

        xi = np.asarray(leja_points[: m + 1], dtype=float)
        if len(xi) != m + 1:
            raise ValueError("Not enough Leja points.")

        # scaling (your convention)
        c = 0.5 * (alpha + beta)
        gamma = 0.25 * (beta - alpha)

        L_hat = self.leja_matrix(xi)
        Lm = c * np.eye(m + 1) + gamma * L_hat

        # scale s for log
        #s = np.max(c*xi+gamma)/2
        s = c
        Wm = (Lm - s * np.eye(m + 1)) / s

        e1 = np.zeros(m + 1, dtype=float)
        e1[0] = 1.0

        dd_log1p = self.log1p_taylor_matvec(Wm, e1, p=self.p)
        dd_log = dd_log1p.copy()
        dd_log[0] += np.log(s)

        return dd_log, xi