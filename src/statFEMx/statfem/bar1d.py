from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from statFEMx.config import Bar1DConfig
from statFEMx.utils.kernels import sqexp, sqexp_derivatives
from statFEMx.utils.linalg import cholesky_inverse, nearest_psd, stable_cholesky, symmetric_part


@dataclass(slots=True)
class ObservationData:
    x_dense: np.ndarray
    true_displacement: np.ndarray
    y_experiment: np.ndarray
    y_obs: np.ndarray
    sensor_indices: np.ndarray
    sensor_coordinates: np.ndarray
    nrep: int
    nsen: int
    epsi: float
    rho_sample: float
    sig_d_sample: float
    l_d_sample: float
    C_e_sample: np.ndarray
    C_d_sample: np.ndarray
    C_e: np.ndarray
    S: float


@dataclass(slots=True)
class HyperparameterResult:
    rho: float
    sig_d: float
    l_d: float
    C_d: np.ndarray
    optimizer_success: bool
    optimizer_message: str


@dataclass(slots=True)
class PosteriorResult:
    mu_u_y: np.ndarray
    C_u_y: np.ndarray
    ci_u_y: np.ndarray
    mu_z: np.ndarray
    C_z: np.ndarray
    ci_z: np.ndarray


def generate_synthetic_observations(
    config: Bar1DConfig,
    *,
    obs_case: str = "nonlinear",
    cal_case: int = 7,
) -> ObservationData:
    nrep_sample = 1000
    n_e_sample = 100
    xx = np.linspace(0.0, config.length, n_e_sample + 1)

    if obs_case == "linear":
        S = 0.0
        u_sample = (config.tip_load / (config.area *
                    config.mean_youngs_modulus)) * xx
    elif obs_case == "nonlinear":
        S = 0.015
        u_sample = (config.tip_load / (config.area * S *
                    config.mean_youngs_modulus)) * (1.0 - np.exp(-S * xx))
    else:
        raise ValueError(f"Unsupported obs_case={obs_case!r}")

    epsi = float(config.observation_noise_levels[0])
    rho_sample = 0.7
    sig_d_sample = 0.9
    l_d_sample = 2.0

    rng_e = np.random.default_rng(4)
    C_e_sample = epsi * np.eye(xx.size)
    e_sample = rng_e.multivariate_normal(
        np.zeros(xx.size), C_e_sample, size=nrep_sample).T

    rng_d = np.random.default_rng(5)
    C_d_sample = sqexp(xx, xx, np.log(sig_d_sample), np.log(l_d_sample))
    d_sample = rng_d.multivariate_normal(
        np.zeros(xx.size), C_d_sample, size=nrep_sample).T

    Y_experiment = np.empty((xx.size, nrep_sample), dtype=float)
    for i in range(nrep_sample):
        Y_experiment[:, i] = rho_sample * \
            u_sample + d_sample[:, i] + e_sample[:, i]

    case_map = {
        1: (1, 11),
        2: (10, 11),
        3: (100, 11),
        4: (1000, 11),
        5: (1, 33),
        6: (10, 33),
        7: (100, 33),
        8: (1000, 33),
        9: (1, 50),
        10: (10, 50),
        11: (100, 50),
        12: (1000, 50),
    }
    if cal_case not in case_map:
        raise ValueError(f"Invalid cal_case={cal_case}")
    nrep, nsen = case_map[cal_case]

    if nsen == 4:
        sen_ind = np.array([20, 40, 60, 80], dtype=np.int32) - 1
    elif nsen == 11:
        sen_ind = np.array([5, 20, 25, 35, 40, 50, 60, 75,
                           80, 90, 101], dtype=np.int32) - 1
    elif nsen == 33:
        sen_ind = np.round(np.linspace(2, 101, nsen)).astype(np.int32) - 1
    else:
        sen_ind = np.arange(1, 101, 2, dtype=np.int32) - 1

    sen_coor = xx[sen_ind]
    y_obs = Y_experiment[sen_ind, :nrep]
    C_e = epsi * np.eye(nsen)

    return ObservationData(
        x_dense=xx,
        true_displacement=u_sample,
        y_experiment=Y_experiment,
        y_obs=y_obs,
        sensor_indices=sen_ind,
        sensor_coordinates=sen_coor,
        nrep=nrep,
        nsen=nsen,
        epsi=epsi,
        rho_sample=rho_sample,
        sig_d_sample=sig_d_sample,
        l_d_sample=l_d_sample,
        C_e_sample=C_e_sample,
        C_d_sample=C_d_sample,
        C_e=C_e,
        S=S,
    )


def build_projection_matrix(config: Bar1DConfig, sensor_coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_coordinates = config.node_coordinates
    nsen = sensor_coordinates.size
    P = np.zeros((nsen, node_coordinates.size), dtype=float)

    for i, x in enumerate(sensor_coordinates):
        if np.isclose(x, node_coordinates[-1]):
            P[i, -1] = 1.0
            continue
        e = np.searchsorted(node_coordinates, x, side="right") - 1
        e = max(0, min(e, config.n_elements - 1))
        x0 = node_coordinates[e]
        x1 = node_coordinates[e + 1]
        xi = (2.0 * x - (x0 + x1)) / (x1 - x0)
        N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)
        P[i, e: e + 2] = N

    return P, P[:, 1:]


def negative_log_likelihood(
    w: np.ndarray,
    C_u_active_pc: np.ndarray,
    P_active: np.ndarray,
    C_e: np.ndarray,
    y_obs: np.ndarray,
    mu_u_active_pc: np.ndarray,
    sensor_coordinates: np.ndarray,
    nrep: int,
) -> tuple[float, np.ndarray]:
    rho = float(w[0])
    log_sig_d = float(w[1])
    log_l_d = float(w[2])

    base = nearest_psd(P_active @ symmetric_part(C_u_active_pc) @ P_active.T)
    Sigma = symmetric_part(
        rho**2 * base + C_e + sqexp(sensor_coordinates, sensor_coordinates, log_sig_d, log_l_d))
    L, Sigma_reg, _ = stable_cholesky(Sigma)
    inv_sigma = np.linalg.solve(
        L.T, np.linalg.solve(L, np.eye(Sigma_reg.shape[0])))

    residuals = y_obs - rho * (P_active @ mu_u_active_pc)[:, None]
    alpha1 = nrep * Sigma.shape[0] * np.log(2.0 * np.pi)
    alpha2 = 2.0 * nrep * np.sum(np.log(np.diag(L)))
    alpha3 = float(np.sum(residuals * (inv_sigma @ residuals)))
    f = 0.5 * (alpha1 + alpha2 + alpha3)

    d_sigma, d_l = sqexp_derivatives(
        sensor_coordinates, sensor_coordinates, log_sig_d, log_l_d)
    grad = np.zeros(3, dtype=float)

    base = nearest_psd(P_active @ symmetric_part(C_u_active_pc) @ P_active.T)
    grad[0] = nrep * rho * 2.0 * np.trace(inv_sigma @ base)
    for i in range(nrep):
        residual = residuals[:, i]
        a1 = (P_active @ mu_u_active_pc).T @ inv_sigma @ residual
        a2 = residual.T @ inv_sigma @ (rho * 2.0 * base @ inv_sigma) @ residual
        a3 = residual.T @ inv_sigma @ (P_active @ mu_u_active_pc)
        grad[0] -= float(a1 + a2 + a3)
    grad[0] *= 0.5

    grad[1] = nrep * np.trace(inv_sigma @ d_sigma)
    for i in range(nrep):
        residual = residuals[:, i]
        grad[1] -= residual.T @ inv_sigma @ d_sigma @ inv_sigma @ residual
    grad[1] *= 0.5

    grad[2] = nrep * np.trace(inv_sigma @ d_l)
    for i in range(nrep):
        residual = residuals[:, i]
        grad[2] -= residual.T @ inv_sigma @ d_l @ inv_sigma @ residual
    grad[2] *= 0.5

    return float(f), grad


def estimate_hyperparameters(
    C_u_active_pc: np.ndarray,
    P_active: np.ndarray,
    C_e: np.ndarray,
    y_obs: np.ndarray,
    mu_u_active_pc: np.ndarray,
    sensor_coordinates: np.ndarray,
    nrep: int,
) -> HyperparameterResult:
    C_u_active_pc = nearest_psd(C_u_active_pc)
    start = np.array([1.0, np.log(0.8), np.log(5.0)], dtype=float)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        return negative_log_likelihood(
            w,
            C_u_active_pc=C_u_active_pc,
            P_active=P_active,
            C_e=C_e,
            y_obs=y_obs,
            mu_u_active_pc=mu_u_active_pc,
            sensor_coordinates=sensor_coordinates,
            nrep=nrep,
        )

    result = minimize(
        objective,
        start,
        jac=True,
        method="BFGS",
        options={"maxiter": 200, "gtol": 1e-6, "disp": False},
    )
    rho, log_sig_d, log_l_d = result.x
    C_d = nearest_psd(
        sqexp(sensor_coordinates, sensor_coordinates, log_sig_d, log_l_d))
    return HyperparameterResult(
        rho=float(rho),
        sig_d=float(np.exp(log_sig_d)),
        l_d=float(np.exp(log_l_d)),
        C_d=C_d,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def compute_posterior(
    config: Bar1DConfig,
    mu_u_active_pc: np.ndarray,
    C_u_active_pc: np.ndarray,
    P_active: np.ndarray,
    y_obs: np.ndarray,
    C_e: np.ndarray,
    C_d: np.ndarray,
    rho: float,
    nrep: int,
    stabilization: float | None = None,
) -> PosteriorResult:
    GDof = config.number_nodes
    active = config.active_dofs
    stab = 0.1 * \
        config.std_youngs_modulus if stabilization is None else float(
            stabilization)

    C_u_active = nearest_psd(np.asarray(
        C_u_active_pc, dtype=float) + stab * np.eye(active.size))
    inv_CdCe = cholesky_inverse(nearest_psd(C_d + C_e))
    inv_Cu = cholesky_inverse(C_u_active)

    M = rho**2 * nrep * (P_active.T @ inv_CdCe @ P_active) + inv_Cu
    C_u_y_active = cholesky_inverse(M)
    rhs = rho * (P_active.T @ inv_CdCe @
                 np.sum(y_obs[:, :nrep], axis=1)) + inv_Cu @ mu_u_active_pc
    mu_u_y_active = C_u_y_active @ rhs

    mu_u_y = np.zeros(GDof, dtype=float)
    C_u_y = np.zeros((GDof, GDof), dtype=float)
    mu_u_y[1:] = rho * mu_u_y_active
    C_u_y[1:, 1:] = rho**2 * C_u_y_active
    ci_u_y = 1.96 * np.sqrt(np.maximum(np.diag(C_u_y), 0.0))

    mu_z = rho * P_active @ mu_u_y_active
    C_z = rho**2 * P_active @ C_u_y_active @ P_active.T + C_d
    ci_z = 1.96 * np.sqrt(np.maximum(np.diag(C_z), 0.0))

    return PosteriorResult(
        mu_u_y=mu_u_y,
        C_u_y=C_u_y,
        ci_u_y=ci_u_y,
        mu_z=mu_z,
        C_z=C_z,
        ci_z=ci_z,
    )
