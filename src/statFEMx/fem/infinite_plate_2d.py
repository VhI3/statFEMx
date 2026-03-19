from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from statFEMx.config import InfinitePlate2DConfig


@dataclass(slots=True)
class InfinitePlate2DSolution:
    node_coordinates: np.ndarray
    element_nodes: np.ndarray
    displacement: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    force: np.ndarray
    reactions: np.ndarray
    von_mises: np.ndarray
    right_edge_nodes: np.ndarray
    left_nodes: np.ndarray
    bottom_nodes: np.ndarray
    youngs_modulus: float
    poisson_ratio: float
    traction: float


def load_matlab_quad_mesh(mesh_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    text = Path(mesh_file).read_text(encoding="utf-8")
    positions = _parse_matlab_matrix(text, "msh.POS")
    quads = _parse_matlab_matrix(text, "msh.QUADS").astype(np.int64)
    node_coordinates = positions[:, :2].astype(float)
    element_nodes = quads[:, :4] - 1
    return node_coordinates, element_nodes


def solve_infinite_plate_linear_2d(
    config: InfinitePlate2DConfig | None = None,
) -> InfinitePlate2DSolution:
    cfg = InfinitePlate2DConfig() if config is None else config
    node_coordinates, element_nodes = load_matlab_quad_mesh(cfg.mesh_file)

    n_nodes = node_coordinates.shape[0]
    n_elements = element_nodes.shape[0]
    gdofs = 2 * n_nodes
    stiffness = lil_matrix((gdofs, gdofs), dtype=float)
    force = np.zeros(gdofs, dtype=float)

    xi_pts, weights = np.polynomial.legendre.leggauss(cfg.quadrature_order)
    constitutive = cfg.plane_strain_matrix

    for element in element_nodes:
        coords = node_coordinates[element, :]
        ke = np.zeros((8, 8), dtype=float)
        for eta, w_eta in zip(xi_pts, weights, strict=True):
            for xi, w_xi in zip(xi_pts, weights, strict=True):
                dndxi = _shape_function_derivatives(xi, eta)
                jacobian = dndxi.T @ coords
                det_j = np.linalg.det(jacobian)
                if det_j <= 0.0:
                    raise ValueError("Encountered non-positive element Jacobian.")
                dndx = dndxi @ np.linalg.inv(jacobian)
                b_matrix = _build_b_matrix(dndx)
                ke += (
                    w_xi
                    * w_eta
                    * cfg.thickness
                    * det_j
                    * (b_matrix.T @ constitutive @ b_matrix)
                )
        edofs = _element_dofs(element)
        stiffness[np.ix_(edofs, edofs)] += ke

    left_nodes = np.flatnonzero(np.isclose(node_coordinates[:, 0], 0.0, atol=cfg.boundary_tol))
    right_nodes = np.flatnonzero(
        np.isclose(node_coordinates[:, 0], cfg.length, atol=cfg.boundary_tol)
    )
    bottom_nodes = np.flatnonzero(
        np.isclose(node_coordinates[:, 1], 0.0, atol=cfg.boundary_tol)
    )

    right_nodes_sorted = right_nodes[np.argsort(node_coordinates[right_nodes, 1])]
    for a, b in zip(right_nodes_sorted[:-1], right_nodes_sorted[1:], strict=True):
        edge_length = abs(node_coordinates[b, 1] - node_coordinates[a, 1])
        traction_load = 0.5 * edge_length * cfg.traction * np.array([1.0, 1.0], dtype=float)
        force[2 * a] += traction_load[0]
        force[2 * b] += traction_load[1]

    prescribed_x = 2 * left_nodes
    prescribed_y = 2 * bottom_nodes + 1
    prescribed = np.unique(np.concatenate([prescribed_x, prescribed_y]))
    active = np.setdiff1d(np.arange(gdofs), prescribed)

    stiffness_csr = stiffness.tocsr()
    u = np.zeros(gdofs, dtype=float)
    u_active = spsolve(stiffness_csr[active][:, active], force[active])
    u[active] = u_active
    reactions = stiffness_csr @ u - force

    ux = u[0::2]
    uy = u[1::2]
    von_mises = _nodal_von_mises(node_coordinates, element_nodes, u, constitutive)

    return InfinitePlate2DSolution(
        node_coordinates=node_coordinates,
        element_nodes=element_nodes,
        displacement=u,
        ux=ux,
        uy=uy,
        force=force,
        reactions=reactions,
        von_mises=von_mises,
        right_edge_nodes=right_nodes_sorted,
        left_nodes=left_nodes,
        bottom_nodes=bottom_nodes,
        youngs_modulus=cfg.youngs_modulus,
        poisson_ratio=cfg.poisson_ratio,
        traction=cfg.traction,
    )


def _parse_matlab_matrix(text: str, variable: str) -> np.ndarray:
    pattern = rf"{re.escape(variable)}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find {variable} in MATLAB mesh file.")
    rows = []
    for line in match.group(1).splitlines():
        stripped = line.strip().rstrip(";")
        if not stripped:
            continue
        rows.append([float(value) for value in stripped.split()])
    return np.asarray(rows, dtype=float)


def _shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [+(1.0 - eta), -(1.0 + xi)],
            [+(1.0 + eta), +(1.0 + xi)],
            [-(1.0 + eta), +(1.0 - xi)],
        ],
        dtype=float,
    )


def _build_b_matrix(dndx: np.ndarray) -> np.ndarray:
    b_matrix = np.zeros((3, 8), dtype=float)
    b_matrix[0, 0::2] = dndx[:, 0]
    b_matrix[1, 1::2] = dndx[:, 1]
    b_matrix[2, 0::2] = dndx[:, 1]
    b_matrix[2, 1::2] = dndx[:, 0]
    return b_matrix


def _element_dofs(element: np.ndarray) -> np.ndarray:
    dofs = np.empty(8, dtype=np.int64)
    dofs[0::2] = 2 * element
    dofs[1::2] = 2 * element + 1
    return dofs


def _nodal_von_mises(
    node_coordinates: np.ndarray,
    element_nodes: np.ndarray,
    displacement: np.ndarray,
    constitutive: np.ndarray,
) -> np.ndarray:
    nodal_vm = np.zeros(node_coordinates.shape[0], dtype=float)
    counts = np.zeros(node_coordinates.shape[0], dtype=np.int64)
    for element in element_nodes:
        coords = node_coordinates[element, :]
        dndxi = _shape_function_derivatives(0.0, 0.0)
        jacobian = dndxi.T @ coords
        dndx = dndxi @ np.linalg.inv(jacobian)
        b_matrix = _build_b_matrix(dndx)
        strain = b_matrix @ displacement[_element_dofs(element)]
        stress = constitutive @ strain
        sigma_xx, sigma_yy, tau_xy = stress
        vm = np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * tau_xy**2)
        for node in element:
            nodal_vm[node] += vm
            counts[node] += 1
    counts[counts == 0] = 1
    return nodal_vm / counts
