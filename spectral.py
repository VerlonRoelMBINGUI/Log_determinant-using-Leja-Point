from __future__ import annotations


class FocalIntervalEstimator:
    """
    Estimate spectral bounds [lambda_min, lambda_max] for SPD/PSD matrices.
    Default: Gershgorin. Optionally you can add eigsh later.
    """
    def __init__(self, min_floor: float = 1e-12):
        self.min_floor = float(min_floor)

    def gershgorin(self, Q: sp.spmatrix) -> tuple[float, float]:
        if not sp.isspmatrix(Q):
            Q = sp.csr_matrix(Q)

        D = Q.diagonal().astype(float)

        # row sum of abs values
        row_abs_sum = np.asarray(np.abs(Q).sum(axis=1)).reshape(-1)  # (n,)
        R = row_abs_sum - np.abs(D)

        lam_min = float(np.min(D - R))
        lam_max = float(np.max(D + R))

        if lam_min <= 0:
            lam_min = self.min_floor

        return lam_min, lam_max