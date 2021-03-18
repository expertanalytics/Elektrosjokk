import time

import dolfin as df
import numpy as np

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

from xalbrain.cellmodels import Cressman

from postfields import (
    Field,
    PointField,
)

from postspec import (
    FieldSpec,
    SaverSpec,
    LoaderSpec,
)

from post import (
    Saver,
    Loader,
)

from pathlib import Path

from typing import (
    Tuple,
    Any,
)


def get_mesh(
    *,
    N: int,
    square_width: float,
    csf_start: float
) -> Tuple[df.Mesh, df.MeshFunction, df.MeshFunction]:
    """Create the mesh [0, 1]^2 cm and corresponding mesh functions.

    At the moment, the interface function is not used.
    """
    mesh = df.UnitSquareMesh(N, N, "crossed")         # 1cm time 1cm

    cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    cell_function.set_all(0)

    interface_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
    interface_function.set_all(0)

    # Make CSF
    csf = df.CompiledSubDomain("x[0] >= x0", x0=csf_start)
    csf.mark(cell_function, 11)

    # make the central square
    x0 = y0 = (1 - square_width) / 2
    x1 = y1 = (1 + square_width) / 2
    square = df.CompiledSubDomain(
        "x[0] >= x0 && x[0] <= x1 && x[1] >= y0 && x[1] <= y1",
        x0=x0, x1=x1, y0=y0, y1=y1
    )
    square.mark(cell_function, 22)
    return mesh, cell_function, interface_function


def get_Kinf(*, square_width: float, K1: float = 4, K2: float = 8):
    x0 = y0 = (1 - square_width) / 2
    x1 = y1 = (1 + square_width) / 2
    square = df.Expression(
        "x[0] >= x0 && x[0] <= x1 && x[1] >= y0 && x[1] <= y1 ? K2 : K1",
        x0=x0, x1=x1, y0=y0, y1=y1, K1=K1, K2=K2, degree=1
    )
    return square


def get_Kinf_circle(*, L: float, K1: float = 4, K2: float = 8) -> df.Expression:
    """L is the radius."""
    Kinf_string = "pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2) <= {L}*{L} ? {K2} : {K1}".format(L=L, K1=K1, K2=K2)
    return df.Expression(Kinf_string, degree=1)


def get_brain(
    *,
    mesh_resolution: int,
    conductivity: float,
    kinf_domain_size: float,
    csf_start: float,
    K1: float = 4,
    K2: float = 8
) -> CardiacModel:
    """
    Create container class for splitting solver parameters

    Arguments:
        N: Mesh resolution.
        conductivity: The conductivity, or rather, the conductivity times a factor.
        kinf_domain_size: The side length of the domain where Kinf = 8. In 1D, this is
            simply the lengt of an interval.
    """
    time_const = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh(
        N=mesh_resolution,
        square_width=kinf_domain_size,
        csf_start=csf_start
    )

    Mi = df.Constant(conductivity)
    Kinf = get_Kinf(square_width=kinf_domain_size, K1=K1, K2=K2)

    # Define cell model
    model_parameters = Cressman.default_parameters()
    model_parameters["Koinf"] = Kinf
    model = Cressman(params=model_parameters)
    brain = CardiacModel(
        mesh,
        time_const,
        M_i=Mi,
        M_e=None,
        cell_models=model,
        cell_domains=cell_function
    )
    return brain


def get_solver(*, brain: CardiacModel) -> SplittingSolver:
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["theta"] = 0.5
    ps["MonodomainSolver"]["linear_solver_type"] = "direct"
    # ps["BidomainSolver"]["use_avg_u_constraint"] = False
    ps["CardiacODESolver"]["scheme"] = "ERK1"

    # ps["MonodomainSolver"]["Chi"] = 1.26e3      # 1/cm -- Dougherty 2015
    # ps["MonodomainSolver"]["Cm"] = 1.0          # muF/cm^2 -- Dougherty 2015

    # # Ratio of Intra to extra cellular conductivity
    # ps["MonodomainSolver"]["lambda"] = brain.intracellular_conductivity()/2.76

    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["cpp_optimize"] = True

    flags = "-O3 -ffast-math -march=native"
    df.parameters["form_compiler"]["cpp_optimize_flags"] = flags

    df.parameters["form_compiler"]["quadrature_degree"] = 1
    solver = SplittingSolver(brain, params=ps)
    return solver


def assign_initial_conditions(*, solver: Any, ic: Any = None) -> None:
    """Assign initial conditions. 

    if initial conditons are not supplied, used defaults.
    """
    brain = solver.model
    model = brain.cell_models

    if ic is None:
        ic = model.initial_conditions()

    vs_, *_ = solver.solution_fields()
    vs_.assign(ic)


def reload_initial_condition(solver: Any, casedir: Path) -> None:
    loader_spec = LoaderSpec(casedir)
    loader = Loader(loader_spec)
    ic = loader.load_initial_condition("v", timestep_index=-10)
    vs_, *_ = solver.solution_fields()
    vs_.assign(ic)
