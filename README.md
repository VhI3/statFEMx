# statFEMx

High-performance Python package for statistical FEM with a FEniCSx-ready workflow, MPI parallel UQ, and publication-quality plotting scripts.

## What it does

- 1D bar benchmark pipeline
- Deterministic solve + uncertainty propagation
- Monte Carlo and PCE-based prior UQ
- Posterior computation for multiple calibration cases
- Automated plotting for prior and posterior summaries

## Repository Layout

- `src/statFEMx/` core package
- `scripts/run_bar1d_full_pipeline.py` full simulation pipeline
- `scripts/plot_bar1d_prior_uq.py` prior UQ figure
- `scripts/plot_bar1d_posterior_grid.py` posterior grid figure
- `docker/Dockerfile` container build recipe

## Docker Quick Start

Build image:

```bash
docker build -t statfemx -f docker/Dockerfile .
```

Create output directory:

```bash
mkdir -p results
```

Run one case (example: nonlinear obs, case 7):

```bash
docker run --rm -it -v "$PWD:/work" statfemx bash -lc 'cd /opt/statfemx && mpiexec -n 4 python3 scripts/run_bar1d_full_pipeline.py --backend analytic --obs-case nonlinear --cal-case 7 --output /work/results/case_7.npz'
```

Run multiple cases:

```bash
for c in 1 2 3 5 6 7; do
  docker run --rm \
    -v "$PWD:/work" \
    statfemx \
    bash -lc "cd /opt/statfemx && mpiexec -n 4 python3 scripts/run_bar1d_full_pipeline.py --backend analytic --obs-case nonlinear --cal-case $c --output /work/results/case_${c}.npz"
done
```

Generate prior UQ plot:

```bash
docker run --rm -it -v "$PWD:/work" statfemx bash -lc 'python3 /work/scripts/plot_bar1d_prior_uq.py /work/results/case_7.npz --output /work/results/bar1d_prior_uq.png'
```

Generate posterior grid plot:

```bash
docker run --rm -it \
  -v "$PWD:/work" \
  statfemx \
  bash -lc 'python3 /work/scripts/plot_bar1d_posterior_grid.py --dir /work/results --output /work/results/bar1d_posterior_grid.png'
```

If matplotlib backend issues appear in container runs, add:

```bash
-e MPLBACKEND=Agg
```

## Local Run (without Docker)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
mpiexec -n 4 python3 scripts/run_bar1d_full_pipeline.py --backend analytic --obs-case nonlinear --cal-case 7 --output results/case_7.npz
```

## Main CLI options

```bash
python scripts/run_bar1d_full_pipeline.py --help
```

Key flags:

- `--backend {analytic,fenicsx}`
- `--obs-case {linear,nonlinear}`
- `--cal-case <int>`
- `--output <path>`

## License

GPL-3.0-or-later
