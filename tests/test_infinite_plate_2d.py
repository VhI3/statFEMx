from __future__ import annotations

import numpy as np

from statFEMx.config import InfinitePlate2DConfig
from statFEMx.fem.infinite_plate_2d import solve_infinite_plate_linear_2d


def test_infinite_plate_linear_solution_respects_symmetry_constraints() -> None:
    sol = solve_infinite_plate_linear_2d(InfinitePlate2DConfig())
    assert np.allclose(sol.ux[sol.left_nodes], 0.0)
    assert np.allclose(sol.uy[sol.bottom_nodes], 0.0)


def test_infinite_plate_linear_solution_has_positive_loaded_edge_displacement() -> None:
    sol = solve_infinite_plate_linear_2d(InfinitePlate2DConfig())
    assert float(np.max(sol.ux[sol.right_edge_nodes])) > 0.0
    assert np.all(np.isfinite(sol.von_mises))
