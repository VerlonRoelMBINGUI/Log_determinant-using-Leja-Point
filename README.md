# Novel Techinque based on Leja points approximation for Log determinant estimation
This repository contains code accompanying the paper: **Novel technique based on Leja Points Approximation for Log-determinant Estimation**
# leja-logdet

Estimate **logdet(Q)=tr(log(Q))**  of large sparse **SPD** matrices using:

- **Newton–Leja interpolation** to approximate the matrix function action `log(Q) v`
- **Hutch++** to estimate `tr(log(Q))` using only matrix–vector products

This is useful in large-scale Gaussian models / GMRFs where log-determinants appear in likelihoods, numerical linear Algebra and Machine Learning.
- We used the data from University of Florida [1]
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


## Cite our paper



## References 
[1] Davis, T.A., Hu, Y., 2011. The university of florida sparse matrix collection. ACM Transactions on Mathematical Software (TOMS) 38, 1–25.

[2] Geiger, S., Lord, G., Tambue, A., 2012. Exponential time integrators for stochastic partial differential equations in 3d reservoir simulation.
Computational Geosciences 16, 323–334.

[3] Deka, P.J., Einkemmer, L., 2022. Exponential integrators for resistive magnetohydrodynamics: Matrix-free leja interpolation and efficient
adaptive time stepping. The Astrophysical Journal Supplement Series 259, 57

