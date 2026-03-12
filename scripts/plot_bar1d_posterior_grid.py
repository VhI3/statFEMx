#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_mpl() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "font.size": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


CASE_MAP = {
    1: (1, 11),
    2: (10, 11),
    3: (100, 11),
    5: (1, 33),
    6: (10, 33),
    7: (100, 33),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("bar1d_posterior_grid.pdf"))
    parser.add_argument("--png-preview", type=Path, default=None)
    args = parser.parse_args()

    setup_mpl()

    cases = [1, 2, 3, 5, 6, 7]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.8))
    axes = axes.ravel()

    for i, case in enumerate(cases):
        data = np.load(args.dir / f"case_{case}.npz")

        x = np.asarray(data["node_coordinates"], dtype=float)
        prior_mean = np.asarray(data["pce_mean"], dtype=float)
        prior_cov = np.asarray(data["pce_covariance"], dtype=float)
        prior_std = np.sqrt(np.maximum(np.diag(prior_cov), 0.0))
        prior_ci = 1.96 * prior_std

        post_mean = np.asarray(data["posterior_mean"], dtype=float)
        post_ci = np.asarray(data["posterior_ci"], dtype=float)

        sensor_x = np.asarray(data["sensor_coordinates"], dtype=float)
        y_obs = np.asarray(data["y_obs"], dtype=float)

        ax = axes[i]

        ax.fill_between(
            x, post_mean - post_ci, post_mean + post_ci,
            color="red", alpha=0.30, linewidth=0.0,
            label=r"$95\%$ CI post." if i == 0 else None
        )
        ax.fill_between(
            x, prior_mean - prior_ci, prior_mean + prior_ci,
            color="blue", alpha=0.30, linewidth=0.0,
            label=r"$95\%$ CI prio." if i == 0 else None
        )
        ax.plot(x, prior_mean, color="blue", linewidth=1.2,
                label=r"$\mu_u^{\mathrm{PC}}$" if i == 0 else None)
        ax.plot(x, post_mean, color="red", linewidth=1.2,
                label=r"$\mu_{u\mid Y}^{\mathrm{PC}}$" if i == 0 else None)

        # Show all observations as vertical point clouds
        for j in range(sensor_x.size):
            ax.plot(
                np.full(y_obs.shape[1], sensor_x[j]),
                y_obs[j, :],
                ".",
                color="black",
                markersize=0.9,
            )

        # Show observed mean as black markers
        ax.plot(
            sensor_x, y_obs.mean(axis=1), "o",
            color="black", markersize=2.5,
            label="obs." if i == 0 else None
        )

        nrep, nsen = CASE_MAP[case]
        ax.set_xlabel(r"$X^h$")
        ax.set_ylabel(r"$u^h(X^h)$")
        ax.tick_params(direction="in")
        ax.grid(False)
        ax.text(
            0.5, -0.28,
            rf"{panel_labels[i]}\;\; n_{{\mathrm{{rep}}}}={nrep},\; n_{{\mathrm{{sen}}}}={nsen}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=14
        )

    axes[0].legend(loc="upper left")
    fig.subplots_adjust(wspace=0.28, hspace=0.42)
    fig.savefig(args.output)

    if args.png_preview is not None:
        fig.savefig(args.png_preview, dpi=300)

    print(f"saved {args.output}")
    if args.png_preview is not None:
        print(f"saved {args.png_preview}")


if __name__ == "__main__":
    main()