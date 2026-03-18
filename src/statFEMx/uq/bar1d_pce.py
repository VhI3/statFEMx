from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
import chaospy as cp

from statFEMx.config import Bar1DConfig
from statFEMx.fem.bar1d import solve_bar_1d
from statFEMx.parallel.ensemble import evaluate_samples_distributed
from statFEMx.utils.linalg import nearest_psd, symmetric_part


@dataclass(slots=True)
class PCEResult:
    expansion_order: int
    collocation_samples: np.ndarray
    collocation_evaluations: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    std: np.ndarray
    output_samples: np.ndarray
    tip_samples: np.ndarray


def run_bar1d_pce(
    config: Bar1DConfig,
    *,
    backend: str = "fenicsx",
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> PCEResult | None:
    lam, zeta = config.lognormal_parameters()

    # Generalized PCE: xi ~ N(0,1), E = exp(lam + zeta*xi)
    xi_dist = cp.Normal(0.0, 1.0)
    expansion = cp.generate_expansion(config.pce_order, xi_dist)

    # Quadrature/projection is much more stable here than random regression
    quad_order = config.pce_order + 2
    nodes, weights = cp.generate_quadrature(
        quad_order, xi_dist, rule="gaussian")
    xi_nodes = np.asarray(nodes[0], dtype=float)
    E_nodes = np.exp(lam + zeta * xi_nodes)

    def evaluator(E: float) -> np.ndarray:
        sol = solve_bar_1d(config, E, backend=backend, comm=MPI.COMM_SELF)
        return sol.displacement

    evaluations = evaluate_samples_distributed(E_nodes, evaluator, comm=comm)
    if comm.rank != 0:
        return None

    assert evaluations is not None

    approx = cp.fit_quadrature(expansion, nodes, weights, evaluations)

    mean = np.asarray(cp.E(approx, xi_dist), dtype=float).reshape(-1)
    covariance = np.asarray(cp.Cov(approx, xi_dist), dtype=float)
    covariance = nearest_psd(symmetric_part(covariance))
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    output_xi = np.asarray(
        xi_dist.sample(config.n_pce_output_samples,
                       rule="random", seed=config.random_seed + 17),
        dtype=float,
    ).reshape(1, -1)

    pce_samples = np.asarray(approx(*output_xi), dtype=float)
    if pce_samples.ndim == 1:
        pce_samples = pce_samples.reshape(1, -1)

    return PCEResult(
        expansion_order=config.pce_order,
        collocation_samples=xi_nodes.copy(),
        collocation_evaluations=evaluations,
        mean=mean,
        covariance=covariance,
        std=std,
        output_samples=pce_samples,
        tip_samples=pce_samples[-1, :],
    )
