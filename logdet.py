from __future__ import annotations
import numpy as np
import scipy.sparse as sp

# from .spectral import FocalIntervalEstimator
# from .leja_action import LejaLogAction
# from .hutchpp import HutchPP

class LogDetEstimatorHutchPPLeja:
    """
    Estimate tr(log(Q)) for SPD Q using:
      - shift-scale alpha_scale = lambda_min
      - A = log(Q/alpha_scale) is PSD
      - tr(log(Q)) = tr(A) + n*log(alpha_scale)
    """
    def __init__(
        self,
        spectral: FocalIntervalEstimator,
        leja_action: LejaLogAction,
        hutchpp: HutchPP,
    ):
        self.spectral = spectral
        self.leja = leja_action
        self.hutchpp = hutchpp

    def _make_matvec_logQt(
        self,
        Q,
        leja_points: np.ndarray,
        a_scaled: float,
        b_scaled: float,
        m_leja: int,
        tol: float,
    ):
        # scaling params for Newton–Leja recurrence
        c = 0.5 * (a_scaled + b_scaled)
        Gamma = 0.25 * (b_scaled - a_scaled)
        leja_m_used = []

        def matvec(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                y, m_used, _ = self.leja.apply(
                    v=X, Q=Q, c=c, Gamma=Gamma,
                    leja_points=leja_points,
                    alpha=a_scaled, beta=b_scaled,
                    m_max=m_leja, tol=tol
                )
                leja_m_used.append(int(m_used))
                return y
            if X.ndim == 2:
                n, k = X.shape
                Y = np.empty((n, k), dtype=float)
                for j in range(k):
                    y, m_used, _ = self.leja.apply(
                        v=X[:, j], Q=Q, c=c, Gamma=Gamma,
                        leja_points=leja_points,
                        alpha=a_scaled, beta=b_scaled,
                        m_max=m_leja, tol=tol
                    )
                    leja_m_used.append(int(m_used))
                    Y[:, j] = y
                return Y
            raise ValueError("X must be 1D or 2D")

        return matvec, leja_m_used

    def estimate(
        self,
        Q,
        leja_points: np.ndarray,
        m_leja: int = 300,
        m_hutchpp: int = 60,
        tol: float = 1e-6,
    ):
        if sp.issparse(Q):
            n = Q.shape[0]
        else:
            Q = np.asarray(Q)
            n = Q.shape[0]

        lam_min, lam_max = self.spectral.gershgorin(Q if sp.issparse(Q) else sp.csr_matrix(Q))

        alpha_scale = float(lam_min)
        if alpha_scale <= 0:
            alpha_scale = 1e-12

        # Qt = Q / alpha_scale
        if sp.issparse(Q):
            Qt = (1.0 / alpha_scale) * Q
        else:
            Qt = (1.0 / alpha_scale) * np.asarray(Q)

        a_scaled = lam_min / alpha_scale  # ~1
        b_scaled = lam_max / alpha_scale

        # enforce PSD log condition safely
        a_scaled = max(1.0, float(a_scaled))
        b_scaled = float(max(b_scaled, a_scaled))

        F_matvec, leja_m_used = self._make_matvec_logQt(
            Q=Qt,
            leja_points=leja_points,
            a_scaled=a_scaled,
            b_scaled=b_scaled,
            m_leja=m_leja,
            tol=tol,
        )

        est_A, info = self.hutchpp.estimate(F_matvec, n=n, m=m_hutchpp)
        est_logQ = est_A + n * np.log(alpha_scale)

        leja_m_used = np.asarray(leja_m_used, dtype=int)
        out = dict(info)
        out.update({
            "lambda_min": float(lam_min),
            "lambda_max": float(lam_max),
            "alpha_scale": float(alpha_scale),
            "a_scaled": float(a_scaled),
            "b_scaled": float(b_scaled),
            "shift_nlogalpha": float(n * np.log(alpha_scale)),
            "used_matvec_budget": int(3 * (m_hutchpp // 3)),
            "leja_m_used": leja_m_used.tolist(),
            "leja_m_max": int(leja_m_used.max()) if leja_m_used.size else 0,
        })

        return float(est_logQ), out