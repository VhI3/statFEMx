from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from mpi4py import MPI

from statFEMx.config import Bar1DConfig
from statFEMx.fem.bar1d import solve_bar_1d
from statFEMx.statfem.bar1d import (
    build_projection_matrix,
    compute_posterior,
    estimate_hyperparameters,
    generate_synthetic_observations,
)
from statFEMx.uq.bar1d_mc import run_bar1d_mc
from statFEMx.uq.bar1d_pce import run_bar1d_pce


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1D statFEM-Recon migration slice.")
    parser.add_argument("--backend", choices=["fenicsx", "analytic"], default="fenicsx")
    parser.add_argument("--obs-case", choices=["linear", "nonlinear"], default="linear")
    parser.add_argument("--cal-case", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("bar1d_results.npz"))
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    cfg = Bar1DConfig()

    det = solve_bar_1d(cfg, cfg.mean_youngs_modulus, backend=args.backend, comm=MPI.COMM_SELF)
    mc = run_bar1d_mc(cfg, backend=args.backend, comm=comm)
    pce = run_bar1d_pce(cfg, backend=args.backend, comm=comm)

    if comm.rank != 0:
        return

    assert mc is not None and pce is not None

    obs = generate_synthetic_observations(cfg, obs_case=args.obs_case, cal_case=args.cal_case)
    P, P_active = build_projection_matrix(cfg, obs.sensor_coordinates)
    hp = estimate_hyperparameters(
        C_u_active_pc=pce.covariance[1:, 1:],
        P_active=P_active,
        C_e=obs.C_e,
        y_obs=obs.y_obs,
        mu_u_active_pc=pce.mean[1:],
        sensor_coordinates=obs.sensor_coordinates,
        nrep=obs.nrep,
    )
    posterior = compute_posterior(
        cfg,
        mu_u_active_pc=pce.mean[1:],
        C_u_active_pc=pce.covariance[1:, 1:],
        P_active=P_active,
        y_obs=obs.y_obs,
        C_e=obs.C_e,
        C_d=hp.C_d,
        rho=hp.rho,
        nrep=obs.nrep,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        node_coordinates=det.node_coordinates,
        deterministic_displacement=det.displacement,
        mc_mean=mc.mean,
        mc_std=mc.std,
        mc_tip_samples=mc.tip_samples,
        pce_mean=pce.mean,
        pce_covariance=pce.covariance,
        pce_tip_samples=pce.tip_samples,
        sensor_coordinates=obs.sensor_coordinates,
        y_obs=obs.y_obs,
        projection_matrix=P,
        rho=hp.rho,
        sig_d=hp.sig_d,
        l_d=hp.l_d,
        posterior_mean=posterior.mu_u_y,
        posterior_covariance=posterior.C_u_y,
        posterior_ci=posterior.ci_u_y,
        measurement_mean=posterior.mu_z,
        measurement_covariance=posterior.C_z,
        measurement_ci=posterior.ci_z,
    )

    print("statFEMx bootstrap 1D pipeline finished.")
    print(f"backend={args.backend}")
    print(f"output={args.output}")
    print(f"rho={hp.rho:.6f}, sig_d={hp.sig_d:.6f}, l_d={hp.l_d:.6f}")
    print(f"MC tip mean={mc.mean[-1]:.6f}, PCE tip mean={pce.mean[-1]:.6f}")


if __name__ == "__main__":
    main()
