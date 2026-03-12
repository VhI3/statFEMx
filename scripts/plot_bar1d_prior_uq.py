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
        "legend.fontsize": 11,
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


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-14)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def ecdf(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("bar1d_prior_uq.pdf"))
    parser.add_argument("--png-preview", type=Path, default=None)
    args = parser.parse_args()

    setup_mpl()

    d = np.load(args.input)

    x = np.asarray(d["node_coordinates"], dtype=float)
    mu = np.asarray(d["pce_mean"], dtype=float)
    cov = np.asarray(d["pce_covariance"], dtype=float)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ci95 = 1.96 * std

    mc_tip = np.asarray(d["mc_tip_samples"], dtype=float)
    pce_tip = np.asarray(d["pce_tip_samples"], dtype=float)

    mu_tip = float(mu[-1])
    sigma_tip = float(std[-1])

    x_pdf = np.linspace(
        min(mc_tip.min(), pce_tip.min(), mu_tip - 4.0 * sigma_tip),
        max(mc_tip.max(), pce_tip.max(), mu_tip + 4.0 * sigma_tip),
        500,
    )
    y_pdf = normal_pdf(x_pdf, mu_tip, sigma_tip)

    x_mc_cdf, y_mc_cdf = ecdf(mc_tip)
    x_pc_cdf, y_pc_cdf = ecdf(pce_tip)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))
    ax1, ax2, ax3, ax4 = axes.flat

    # (a) prior mean and CI
    ax1.fill_between(
        x, mu - ci95, mu + ci95,
        color="blue", alpha=0.45, linewidth=0.0, label=r"$95\%$ CI"
    )
    ax1.plot(x, mu, color="blue", linewidth=1.2, label=r"$\mu_u^{\mathrm{PC}}$")
    ax1.set_xlabel(r"$X^h$")
    ax1.set_ylabel(r"$u^h(X^h)$")
    ax1.legend(loc="lower right")
    ax1.text(0.5, -0.28, r"(a)", transform=ax1.transAxes, ha="center", va="top", fontsize=16)

    # (b) histogram + PDF
    ax2.hist(
        mc_tip, bins=34, density=True,
        color="#6baed6", edgecolor="black", linewidth=0.5, alpha=0.9, label="Hist."
    )
    ax2.plot(x_pdf, y_pdf, color="blue", linewidth=1.5, label="PDF")
    ax2.set_xlabel(r"$u^h(L)$")
    ax2.set_ylabel(r"$f\,(u^h(L))$")
    ax2.legend(loc="upper right")
    ax2.text(0.5, -0.28, r"(b)", transform=ax2.transAxes, ha="center", va="top", fontsize=16)

    # (c) CDF comparison
    ax3.plot(x_mc_cdf, y_mc_cdf, color="blue", linewidth=1.3, label="MC")
    ax3.plot(x_pc_cdf, y_pc_cdf, color="red", linewidth=1.3, linestyle=(0, (5, 3)), label="PC")
    ax3.set_xlabel(r"$u^h(L)$")
    ax3.set_ylabel(r"$F\,(u^h(L))$")
    ax3.set_ylim(0.0, 1.0)
    ax3.legend(loc="lower right")
    ax3.text(0.5, -0.28, r"(c)", transform=ax3.transAxes, ha="center", va="top", fontsize=16)

    # (d) PDF comparison
    ax4.hist(mc_tip, bins=45, density=True, histtype="step", linewidth=1.3, color="blue", label="MC")
    ax4.hist(pce_tip, bins=45, density=True, histtype="step", linewidth=1.3,
             linestyle=(0, (5, 3)), color="red", label="PC")
    ax4.set_xlabel(r"$u^h(L)$")
    ax4.set_ylabel(r"$f\,(u^h(L))$")
    ax4.legend(loc="upper right")
    ax4.text(0.5, -0.28, r"(d)", transform=ax4.transAxes, ha="center", va="top", fontsize=16)

    for ax in axes.flat:
        ax.tick_params(direction="in")
        ax.grid(False)

    fig.subplots_adjust(wspace=0.32, hspace=0.42)
    fig.savefig(args.output)

    if args.png_preview is not None:
        fig.savefig(args.png_preview, dpi=300)

    print(f"saved {args.output}")
    if args.png_preview is not None:
        print(f"saved {args.png_preview}")


if __name__ == "__main__":
    main()