from __future__ import annotations

import numpy as np


def _pairwise_sqdist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(x1, dtype=float)).reshape(-1, 1)
    b = np.atleast_2d(np.asarray(x2, dtype=float)).reshape(-1, 1)
    return (a - b.T) ** 2


def sqexp(x1: np.ndarray, x2: np.ndarray, log_sigma: float, log_l: float) -> np.ndarray:
    r2 = _pairwise_sqdist(x1, x2)
    return np.exp(2.0 * log_sigma) * np.exp(-0.5 * r2 * np.exp(-2.0 * log_l))


def sqexp_derivatives(x1: np.ndarray, x2: np.ndarray, log_sigma: float, log_l: float) -> tuple[np.ndarray, np.ndarray]:
    r2 = _pairwise_sqdist(x1, x2)
    K = sqexp(x1, x2, log_sigma, log_l)
    d_sigma = 2.0 * K
    d_l = K * r2 * np.exp(-2.0 * log_l)
    return d_sigma, d_l
