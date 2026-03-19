from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from statFEMx.config import InfinitePlate2DConfig
from statFEMx.fem.infinite_plate_2d import solve_infinite_plate_linear_2d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 2D quarter infinite-plate-with-hole linear-elastic benchmark."
    )
    parser.add_argument("--youngs-modulus", type=float, default=200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.25)
    parser.add_argument("--traction", type=float, default=100.0)
    parser.add_argument(
        "--mesh-file",
        type=Path,
        default=Path("data/infinite_plate_2d/Mesh_infPlate.m"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/infinite_plate_2d_linear.npz"),
    )
    args = parser.parse_args()

    cfg = InfinitePlate2DConfig(
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        traction=args.traction,
        mesh_file=args.mesh_file,
    )
    solution = solve_infinite_plate_linear_2d(cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        node_coordinates=solution.node_coordinates,
        element_nodes=solution.element_nodes,
        displacement=solution.displacement,
        ux=solution.ux,
        uy=solution.uy,
        force=solution.force,
        reactions=solution.reactions,
        von_mises=solution.von_mises,
        left_nodes=solution.left_nodes,
        right_edge_nodes=solution.right_edge_nodes,
        bottom_nodes=solution.bottom_nodes,
        youngs_modulus=solution.youngs_modulus,
        poisson_ratio=solution.poisson_ratio,
        traction=solution.traction,
    )

    print("statFEMx 2D infinite-plate linear-elastic solve finished.")
    print(f"output={args.output}")
    print(f"max|ux|={np.max(np.abs(solution.ux)):.6e}")
    print(f"max|uy|={np.max(np.abs(solution.uy)):.6e}")
    print(f"max von Mises={np.max(solution.von_mises):.6e}")


if __name__ == "__main__":
    main()
