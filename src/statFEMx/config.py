from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(slots=True)
class Bar1DConfig:
    length: float = 100.0
    tip_load: float = 800.0
    area: float = 20.0
    n_elements: int = 30
    mean_youngs_modulus: float = 200.0
    std_youngs_modulus: float = 15.0
    n_mc: int = 2000
    pce_order: int = 10 
    n_pce_samples: int | None = None
    n_pce_output_samples: int = 200
    random_seed: int = 3
    observation_noise_levels: tuple[float, ...] = (0.004, 0.04, 0.4, 4.0)
    # observation_noise_levels: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)  # for testing

    def __post_init__(self) -> None:
        if self.n_pce_samples is None:
            self.n_pce_samples = 3 * self.pce_order

    @property
    def number_nodes(self) -> int:
        return self.n_elements + 1

    @property
    def node_coordinates(self) -> np.ndarray:
        return np.linspace(0.0, self.length, self.number_nodes)

    @property
    def active_dofs(self) -> np.ndarray:
        return np.arange(1, self.number_nodes, dtype=np.int32)

    def exact_linear_displacement(self, youngs_modulus: float | None = None) -> np.ndarray:
        E = self.mean_youngs_modulus if youngs_modulus is None else float(youngs_modulus)
        return (self.tip_load / (self.area * E)) * self.node_coordinates

    def lognormal_parameters(self) -> tuple[float, float]:
        mu = self.mean_youngs_modulus
        sigma = self.std_youngs_modulus
        lam = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
        zeta = np.sqrt(np.log(1.0 + sigma**2 / mu**2))
        return float(lam), float(zeta)


@dataclass(slots=True)
class InfinitePlate2DConfig:
    length: float = 4.0
    height: float = 4.0
    thickness: float = 1.0
    hole_radius: float = 0.4
    youngs_modulus: float = 200.0
    poisson_ratio: float = 0.25
    traction: float = 100.0
    quadrature_order: int = 2
    boundary_tol: float = 1e-10
    mesh_file: Path | None = None

    def __post_init__(self) -> None:
        if self.mesh_file is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.mesh_file = repo_root / "data" / "infinite_plate_2d" / "Mesh_infPlate.m"

    @property
    def lame_lambda(self) -> float:
        nu = self.poisson_ratio
        e = self.youngs_modulus
        return float(nu * e / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    @property
    def lame_mu(self) -> float:
        e = self.youngs_modulus
        nu = self.poisson_ratio
        return float(e / (2.0 * (1.0 + nu)))

    @property
    def plane_strain_matrix(self) -> np.ndarray:
        nu = self.poisson_ratio
        scale = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return scale * np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, 0.5 - nu],
            ],
            dtype=float,
        )
