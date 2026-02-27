# Novel Techinque based on Leja points approximation for Log determinant estimation
This repository contains code related to the implementation of the log determinant
# leja-logdet

Estimate **logdet(Q)=tr(log(Q))**  of large sparse **SPD** matrices using:

- **Newton–Leja interpolation** to approximate the matrix function action `log(Q) v`
- **Hutch++** to estimate `tr(log(Q))` using only matrix–vector products

This is useful in large-scale Gaussian models / GMRFs where log-determinants appear in likelihoods.

---

## Features

- Works with **sparse** matrices (`scipy.sparse`) and supports **matrix-free** `matvec` callables
- Gershgorin-based spectral interval estimate `[λ_min, λ_max]`
- PSD-safe shifting:
  \[
  \mathrm{tr}(\log(Q)) = \mathrm{tr}(\log(Q/\alpha)) + n\log(\alpha), \quad \alpha \le \lambda_{\min}(Q)
  \]
- Adaptive Leja stopping criterion:
  \[
  e_m = |d_m|\;\|w_m\|_2 \le \texttt{tol}
  \]
