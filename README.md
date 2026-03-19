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
- `scripts/run_infinite_plate_2d_linear.py` 2D quarter infinite-plate linear-elastic example
- `scripts/plot_bar1d_prior_uq.py` prior UQ figure
- `scripts/plot_bar1d_posterior_grid.py` posterior grid figure
- `notebooks/` Jupyter notebook versions of the examples
- `data/infinite_plate_2d/` vendored 2D benchmark mesh and geometry assets
- `docker/Dockerfile` container build recipe

## Docker Quick Start

Build image:

```bash
docker build -t statfemx -f docker/Dockerfile .
```

The Docker image is based on `dolfinx/dolfinx:stable` and includes the project package, plotting stack, test tools, and JupyterLab.

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

Run the deterministic 2D linear-elastic benchmark:

```bash
python3 scripts/run_infinite_plate_2d_linear.py --output results/infinite_plate_2d_linear.npz
```

## Notebook Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
python -m ipykernel install --user --name statfemx
jupyter lab
```

Notebook examples:

- `notebooks/01_bar1d_full_pipeline.ipynb`
- `notebooks/02_bar1d_prior_uq_plot.ipynb`
- `notebooks/03_bar1d_posterior_grid.ipynb`
- `notebooks/04_infinite_plate_2d_linear.ipynb`

## Notebook Run In Docker

The Docker image already includes JupyterLab and the notebook dependencies.

Before starting Jupyter, make sure the output directories are writable by your host user:

```bash
mkdir -p results results_notebook
```

Start JupyterLab from the container with your host UID/GID and an inline matplotlib backend:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -p 8888:8888 \
  -v "$PWD:/work" \
  -e MPLBACKEND=module://matplotlib_inline.backend_inline \
  statfemx \
  bash -lc 'cd /work && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser'
```

Then open the URL printed by Jupyter in your browser.

Notes:

- notebooks should be opened from `/work/notebooks`
- `--user "$(id -u):$(id -g)"` keeps notebook saves owned by your host user instead of `root` or `nobody`
- `MPLBACKEND=module://matplotlib_inline.backend_inline` ensures `plt.show()` renders inside the notebook
- if your existing `results/` directory was created by an older container run and is owned by `nobody`, either rename it and recreate it or save notebook outputs to `results_notebook/`
- generated `.npz` and `.png` outputs will be written to your mounted local `results/` or `results_notebook/` directory
- the image default `MPLBACKEND=Agg` is still useful for non-interactive script runs; the `docker run` command above overrides it for notebook use

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
