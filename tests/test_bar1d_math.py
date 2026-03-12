from __future__ import annotations

import numpy as np

from statFEMx.config import Bar1DConfig
from statFEMx.fem.bar1d import solve_bar_1d
from statFEMx.statfem.bar1d import build_projection_matrix, generate_synthetic_observations
from statFEMx.utils.linalg import is_positive_definite, nearest_pd


def test_analytic_bar_solution_matches_exact_formula() -> None:
    cfg = Bar1DConfig()
    sol = solve_bar_1d(cfg, cfg.mean_youngs_modulus, backend="analytic")
    expected = cfg.exact_linear_displacement()
    assert np.allclose(sol.displacement, expected)


def test_projection_matrix_row_sums_are_one() -> None:
    cfg = Bar1DConfig()
    obs = generate_synthetic_observations(cfg, obs_case="linear", cal_case=7)
    P, _ = build_projection_matrix(cfg, obs.sensor_coordinates)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_nearest_pd_repairs_indefinite_matrix() -> None:
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert not is_positive_definite(A)
    B = nearest_pd(A)
    assert is_positive_definite(B)
    assert np.allclose(B, B.T)
