from __future__ import annotations

from dataclasses import dataclass
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
