from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from mpi4py import MPI

from uuid import uuid4

from statFEMx.config import Bar1DConfig

try:
    from petsc4py import PETSc
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    import ufl

    HAS_DOLFINX = True
except Exception:  # pragma: no cover - optional import
    HAS_DOLFINX = False


Backend = Literal["fenicsx", "analytic"]


@dataclass(slots=True)
class Bar1DSolution:
    node_coordinates: np.ndarray
    displacement: np.ndarray
    youngs_modulus: float
    backend: str


def solve_bar_1d(
    config: Bar1DConfig,
    youngs_modulus: float,
    *,
    backend: Backend = "fenicsx",
    comm: MPI.Intracomm | None = None,
    petsc_options: dict[str, object] | None = None,
) -> Bar1DSolution:
    if backend == "analytic":
        x = config.node_coordinates.copy()
        u = (config.tip_load / (config.area * float(youngs_modulus))) * x
        return Bar1DSolution(x, u, float(youngs_modulus), backend="analytic")

    if not HAS_DOLFINX:
        raise ImportError(
            "FEniCSx/petsc4py is not available. Use backend='analytic' for smoke tests.")

    if comm is None:
        comm = MPI.COMM_SELF

    domain = mesh.create_interval(
        comm, config.n_elements, [0.0, config.length])
    V = fem.functionspace(domain, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E = fem.Constant(domain, PETSc.ScalarType(float(youngs_modulus)))
    A = fem.Constant(domain, PETSc.ScalarType(float(config.area)))
    traction = fem.Constant(domain, PETSc.ScalarType(float(config.tip_load)))

    a_form = E * A * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    fdim = domain.topology.dim - 1
    right_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], config.length))
    facet_values = np.full(len(right_facets), 1, dtype=np.int32)
    facet_tags = mesh.meshtags(domain, fdim, right_facets, facet_values)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    l_form = traction * v * ds(1)

    left_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs, V)

    options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    }

    prefix = f"bar1d_{uuid4().hex[:8]}_"

    problem = LinearProblem(
        a_form,
        l_form,
        bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options=options,
    )
    uh = problem.solve()

    coords = V.tabulate_dof_coordinates()[:, 0]
    values = uh.x.array.real.copy()
    order = np.argsort(coords)
    return Bar1DSolution(coords[order], values[order], float(youngs_modulus), backend="fenicsx")
