"""
Class-separability metrics and the MSE-HTC corridor of the paper.

The scatter matrices and the HTC follow `Utils/elm_classifier.py` (Eqs. 4-8 of
the paper); the corridor and the spectral description of the test MSE follow
Eqs. (12)-(14) and (25)-(27), for K classes with a normalised discriminant
spectrum p_i = lambda_i / HTC.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit

K = 10                  # number of classes
MSE_CHANCE = (K - 1) / K**2   # random-choice floor (K-1)/K^2


def scatter_matrices(H, y):
    """Within-class and between-class scatter matrices S_W and S_B (Eqs. 6-7)."""
    H = np.asarray(H, dtype=float)
    mu = H.mean(axis=0)
    n = H.shape[1]
    S_W = np.zeros((n, n))
    S_B = np.zeros((n, n))
    for c in np.unique(y):
        Hc = H[y == c]
        mu_c = Hc.mean(axis=0)
        S_W += (Hc - mu_c).T @ (Hc - mu_c)
        d = (mu_c - mu).reshape(-1, 1)
        S_B += Hc.shape[0] * (d @ d.T)
    return S_W, S_B


def hotelling_trace(S_W, S_B, eps=1e-8):
    """HTC = tr[(S_W + eps I)^-1 S_B] (Eq. 8)."""
    return float(np.trace(np.linalg.solve(S_W + eps * np.eye(len(S_W)), S_B)))


def discriminant_spectrum(S_W, S_B, eps=1e-8, k=K - 1):
    """The k largest eigenvalues lambda_i of (S_W + eps I)^-1 S_B, clipped at 0."""
    lam = eigh(S_B, S_W + eps * np.eye(len(S_W)), eigvals_only=True)
    return np.clip(np.sort(lam)[::-1][:k], 0, None)


def normalised_spectra(lam):
    """p_i = lambda_i / sum_i lambda_i for each row of lam (Eq. 25)."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    return lam / lam.sum(axis=1, keepdims=True)


def mse_isotropic(htc):
    """Lower bound of the training MSE, isotropic spectrum lambda_i = HTC/(K-1) (Eq. 13)."""
    return (K - 1)**2 / (K**2 * (K - 1 + np.asarray(htc, dtype=float)))


def mse_spectral(P, htc):
    """Ensemble-averaged MSE <(1/K^2) sum_i 1/(1 + p_i^(s) HTC)>_s (Eqs. 14, 26).

    P holds one normalised spectrum per row (one row per operating point); the
    average over the rows is taken of the MSE, not of the spectrum.
    """
    P = normalised_spectra(P)
    H = np.atleast_1d(np.asarray(htc, dtype=float))[:, None, None]
    return np.sum(1.0 / (1.0 + H * P[None, :, :]), axis=2).mean(axis=1) / K**2


def eta(P):
    """eta = (1/(K-1)) sum_i 1/p_i of the mean normalised spectrum (Eq. 27)."""
    p_mean = normalised_spectra(P).mean(axis=0)
    return float(np.mean(1.0 / p_mean))


def fit_offset(P, htc, mse):
    """Fit MSE(HTC) = mse_spectral(P, HTC) + d with d the only free parameter.

    Least squares in log10(MSE), as for Fig. 4 of the paper. Returns d, its
    standard error, and the rms residual in decades with and without d.
    """
    htc = np.asarray(htc, dtype=float)
    log_mse = np.log10(np.asarray(mse, dtype=float))
    popt, pcov = curve_fit(lambda h, d: np.log10(mse_spectral(P, h) + d),
                           htc, log_mse, p0=[3e-3], bounds=(-2e-2, 5e-2), maxfev=20000)
    d = float(popt[0])
    rms = float(np.sqrt(np.mean((log_mse - np.log10(mse_spectral(P, htc) + d))**2)))
    rms0 = float(np.sqrt(np.mean((log_mse - np.log10(mse_spectral(P, htc)))**2)))
    return d, float(np.sqrt(pcov[0, 0])), rms, rms0
