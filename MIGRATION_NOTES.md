# Migration notes: first executable slice

This bootstrap intentionally starts with the **1D tension bar**, because it lets us lock down:

- FEM semantics in FEniCSx,
- MPI orchestration for many-sample runs,
- the statFEM likelihood and posterior algebra,
- containerized reproducibility.

## Numerical alignment choices

The following are deliberately mirrored from the MATLAB repository:

- `L = 100`, `f_bar = 800`, `A = 20`, `nElm = 30`,
- log-normal Young's modulus with `mu_E = 200`, `sig_E = 15`,
- Monte Carlo sample count `nMC = 2000`,
- PCE order `P_PCE = 8`,
- synthetic-observation seeds `(4, 5)`,
- synthetic-observation defaults `rho=1.2`, `sig_d=0.9`, `l_d=4`,
- calibration case `7` as the default nonlinear data path.

## Performance stance for this first slice

The current code optimizes first for **sample-parallel throughput**:

- rank-local FEniCSx solves use `MPI.COMM_SELF`,
- `MPI.COMM_WORLD` is used to distribute the ensemble,
- root gathers only once per ensemble stage.

That is the right starting point for the 1D benchmark and for your stated priority of fastest many-sample UQ runs.

## Immediate next implementation tasks

- load MATLAB reference outputs and write regression tests,
- add kernel-density/PDF/CDF post-processing,
- expose PETSc options through a config file,
- build the 2D linear-elastic plate-with-hole preprocessing and solve path,
- then add the St. Venant–Kirchhoff branch.
