from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI

from statFEMx.config import Bar1DConfig
from statFEMx.fem.bar1d import solve_bar_1d
from statFEMx.parallel.ensemble import evaluate_samples_distributed


@dataclass(slots=True)
class MonteCarloResult:
    youngs_modulus_samples: np.ndarray
    displacements: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    tip_samples: np.ndarray


def run_bar1d_mc(
    config: Bar1DConfig,
    *,
    backend: str = "fenicsx",
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> MonteCarloResult | None:
    lam, zeta = config.lognormal_parameters()
    if comm.rank == 0:
        rng = np.random.default_rng(config.random_seed)
        xi = rng.standard_normal(config.n_mc)
        E_samples = np.exp(lam + zeta * xi)
    else:
        E_samples = None
    E_samples = comm.bcast(E_samples, root=0)

    def evaluator(E: float) -> np.ndarray:
        sol = solve_bar_1d(config, E, backend=backend, comm=MPI.COMM_SELF)
        return sol.displacement

    displacements = evaluate_samples_distributed(E_samples, evaluator, comm=comm)
    if comm.rank != 0:
        return None

    assert displacements is not None
    return MonteCarloResult(
        youngs_modulus_samples=np.asarray(E_samples, dtype=float),
        displacements=displacements,
        mean=displacements.mean(axis=0),
        std=displacements.std(axis=0, ddof=0),
        tip_samples=displacements[:, -1],
    )
