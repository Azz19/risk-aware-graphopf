from __future__ import annotations

import numpy as np


def _validate_std(std: np.ndarray) -> np.ndarray:
    std = np.asarray(std, dtype=float)
    if std.ndim != 1:
        raise ValueError("std must be one-dimensional")
    if np.any(std < 0):
        raise ValueError("std must be non-negative")
    return std


def _factor_correlation(correlation: np.ndarray) -> np.ndarray:
    correlation = np.asarray(correlation, dtype=float)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be square")
    if not np.allclose(correlation, correlation.T, atol=1e-10):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
        raise ValueError("correlation diagonal must equal one")
    eigval, eigvec = np.linalg.eigh(correlation)
    if eigval.min() < -1e-8:
        raise ValueError("correlation must be positive semidefinite")
    eigval = np.clip(eigval, 0.0, None)
    return eigvec @ np.diag(np.sqrt(eigval))


def exponential_correlation(distance: np.ndarray, length_scale: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    if np.any(distance < 0):
        raise ValueError("distance must be non-negative")
    corr = np.exp(-distance / length_scale)
    np.fill_diagonal(corr, 1.0)
    return corr


def sample_gaussian(rng, n_samples: int, std: np.ndarray, correlation=None) -> np.ndarray:
    std = _validate_std(std)
    z = rng.standard_normal((n_samples, std.size))
    if correlation is not None:
        z = z @ _factor_correlation(correlation).T
    return z * std


def sample_student_t(rng, n_samples: int, std: np.ndarray, df: float, correlation=None) -> np.ndarray:
    # Common chi-square scaling yields an elliptical multivariate Student-t.
    # sqrt((df-2)/df) standardizes each marginal to unit variance.
    if df <= 2:
        raise ValueError("df must exceed 2 for finite variance")
    std = _validate_std(std)
    z = rng.standard_normal((n_samples, std.size))
    if correlation is not None:
        z = z @ _factor_correlation(correlation).T
    chi2 = rng.chisquare(df, size=(n_samples, 1))
    t = z / np.sqrt(chi2 / df)
    t *= np.sqrt((df - 2.0) / df)
    return t * std
