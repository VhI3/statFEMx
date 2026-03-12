from __future__ import annotations

import numpy as np


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def symmetric_part(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def is_positive_definite(A: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(symmetric_part(np.asarray(A, dtype=float)))
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_pd(A: np.ndarray) -> np.ndarray:
    """Return a numerically positive-definite matrix close to ``A``.

    This is a NumPy implementation of the Higham / D'Errico nearest-SPD
    construction, matching the fallback the user requested.
    """
    A = symmetric_part(np.asarray(A, dtype=float))
    B = symmetric_part(A)
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    A2 = symmetric_part(B + H)
    A3 = symmetric_part(A2)

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0], dtype=float)
    k = 1
    while not is_positive_definite(A3):
        mineig = float(np.min(np.real(np.linalg.eigvals(A3))))
        A3 = A3 + I * (-mineig * (k**2) + spacing)
        A3 = symmetric_part(A3)
        k += 1
    return A3


def nearest_psd(A: np.ndarray, min_eig: float = 1e-10) -> np.ndarray:
    A = symmetric_part(np.asarray(A, dtype=float))
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, min_eig)
    return (evecs * evals) @ evecs.T


def stable_cholesky(A: np.ndarray, jitter: float = 1e-12, max_tries: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    A_sym = symmetric_part(np.asarray(A, dtype=float))
    scale = max(1.0, float(np.max(np.abs(np.diag(A_sym)))) if A_sym.size else 1.0)
    current = jitter * scale
    last_error = None
    for _ in range(max_tries):
        try:
            A_reg = A_sym + current * np.eye(A_sym.shape[0])
            L = np.linalg.cholesky(A_reg)
            return L, A_reg, current
        except np.linalg.LinAlgError as exc:
            last_error = exc
            current *= 10.0

    # First use the user's requested Higham nearest-PD fallback.
    A_pd = nearest_pd(A_sym)
    try:
        L = np.linalg.cholesky(A_pd)
        return L, A_pd, current
    except np.linalg.LinAlgError:
        pass

    # Final fallback: explicit eigenvalue clipping.
    A_psd = nearest_psd(A_sym, min_eig=max(current, 1e-10))
    try:
        L = np.linalg.cholesky(A_psd)
        return L, A_psd, current
    except np.linalg.LinAlgError:
        raise last_error if last_error is not None else np.linalg.LinAlgError("stable_cholesky failed")


def cholesky_inverse(A: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    L, _, _ = stable_cholesky(A, jitter=jitter)
    return cholesky_solve(L, np.eye(A.shape[0]))
