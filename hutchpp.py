from __future__ import annotations
import numpy as np

class HutchPP:
    """
    Hutch++ trace estimator for PSD matrices given as matvec.
    Uses m_S=m_Q=m_G=floor(m/3).
    """
    def __init__(self, rng: int | None = 42):
        self.rng = np.random.default_rng(rng)

    @staticmethod
    def _rademacher(rng: np.random.Generator, shape):
        return rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1  # ±1

    def estimate(self, matvec, n: int, m: int = 60):
        m = int(m)
        if m < 6:
            raise ValueError("m must be >= 6 for Hutch++.")
        m3 = m // 3
        m_S = m_Q = m_G = m3

        S = self._rademacher(self.rng, (n, m_S))

        # Y = F S
        try:
            Y = matvec(S)
        except Exception:
            Y = np.column_stack([matvec(S[:, i]) for i in range(m_S)])

        Q, _ = np.linalg.qr(Y, mode="reduced")
        r = Q.shape[1]
        if r == 0:
            return 0.0, {"m_S": m_S, "m_Q": m_Q, "m_G": m_G, "trace_proj": 0.0, "trace_resid": 0.0}

        # trace_proj = tr(Q^T F Q)
        try:
            FQ = matvec(Q)
        except Exception:
            FQ = np.column_stack([matvec(Q[:, i]) for i in range(r)])
        trace_proj = float(np.sum(Q * FQ))

        # Hutchinson on residual
        G = self._rademacher(self.rng, (n, m_G))
        Z = G - Q @ (Q.T @ G)
        try:
            FZ = matvec(Z)
        except Exception:
            FZ = np.column_stack([matvec(Z[:, i]) for i in range(m_G)])
        trace_resid = float(np.sum(Z * FZ) / m_G)

        est = trace_proj + trace_resid
        info = {"m_S": m_S, "m_Q": m_Q, "m_G": m_G, "trace_proj": trace_proj, "trace_resid": trace_resid}
        return est, info