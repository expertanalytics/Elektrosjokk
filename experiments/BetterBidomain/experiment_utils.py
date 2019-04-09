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


def get_mesh(dimension: int, N: int) -> df.Mesh:
    """Create the mesh [0, 1]^d cm."""
    if dimension == 1:
        mesh = df.UnitIntervalMesh(N)
    elif dimension == 2:
        mesh = df.UnitSquareMesh(N, N)         # 1cm time 1cm
    elif dimension == 3:
        mesh = df.UnitCubeMesh(N, N, N)       # 1cm time 1cm
    return mesh


def get_Kinf(dimension: int, L: float, K1: float, K2: float) -> df.Expression:
    """Assume unit square. """
    if dimension == 1:
        kinf_str = "x[0] < {d1} || x[0] > {d2} ? {K1} : {K2}"
    elif dimension == 2:
        kinf_str = "x[0] < {d1} || x[0] > {d2}"
        kinf_str += " || x[1] < {d1} || x[1] > {d2} ? {K1}: {K2}"
    elif dimension == 3:
        kinf_str = "x[0] < {d1} || x[0] > {d2}"
        kinf_str += " || x[1] < {d1} || x[1] > {d2}"
        kinf_str += " || x[2] < {d1} || x[2] > {d2} ? {K1} : {K2}"
    else:
        assert False, "Expected dimension in {1, 2, 3}, got {}".format(dimension)
    Kinf = df.Expression(kinf_str.format(d1=0.5 - L/2, d2=0.5 + L/2, K1=K1, K2=K2), degree=1)
    return Kinf


def get_brain(
        dimension: int,
        N: int,
        conductivity: float,
        Kinf_domain_size: float,
        K1: float = 4,
        K2: float = 8
) -> CardiacModel:
    """
    Create container class for splitting solver parameters

    Arguments:
        dimension: The topological dimension of the mesh.
        N: Mesh resolution.
        conductivity: The conductivity, or rather, the conductivity times a factor.
        Kinf_domain_size: The side length of the domain where Kinf = 8. In 1D, this is
            simply the lengt of an interval.
    """
    mesh = get_mesh(dimension, N)
    Mi = df.Constant(conductivity)
    time_const = df.Constant(0)
    Kinf = get_Kinf(dimension, Kinf_domain_size, K1, K2)

    model_parameters = Cressman.default_parameters()
    model_parameters["Koinf"] = Kinf
    model = Cressman(params=model_parameters)
    brain = CardiacModel(
        mesh,
        time_const,
        M_i=Mi,
        M_e=None,
        cell_models=model,
    )
    return brain


def get_solver(brain: CardiacModel) -> SplittingSolver:
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["theta"] = 0.5
    ps["MonodomainSolver"]["linear_solver_type"] = "direct"
    ps["BidomainSolver"]["use_avg_u_constraint"] = False

    # ps["ode_solver_choice"] = "BetterODESolver"
    ps["CardiacODESolver"]["scheme"] = "RK4"

    ps["MonodomainSolver"]["Chi"] = 1.26e3      # 1/cm -- Dougherty 2015
    ps["MonodomainSolver"]["Cm"] = 1.0          # muF/cm^2 -- Dougherty 2015

    # Ratio of Intra to extra cellular conductivity
    ps["MonodomainSolver"]["lambda"] = brain.intracellular_conductivity()/2.76

    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["cpp_optimize"] = True

    flags = "-O3 -ffast-math -march=native"
    df.parameters["form_compiler"]["cpp_optimize_flags"] = flags

    df.parameters["form_compiler"]["quadrature_degree"] = 1
    solver = SplittingSolver(brain, params=ps)
    return solver


def assign_initial_conditions(solver: Any) -> None:
    brain = solver.model
    model = brain.cell_models
    vs_, *_ = solver.solution_fields()
    vs_.assign(model.initial_conditions())


def reload_initial_condition(solver: Any, casedir: Path) -> None:
    loader_spec = LoaderSpec(casedir)
    loader = Loader(loader_spec)
    ic = loader.load_initial_condition("v", timestep_index=-10)
    vs_, *_ = solver.solution_fields()
    vs_.assign(ic)


def get_points(dimension: int, num_points: int) -> np.ndarray:
    _npj = num_points*1j
    if dimension == 1:
        numbers = np.mgrid[0:1:_npj]
        return np.vstack(map(lambda x: x.ravel(), numbers)).reshape(-1, dimension)
    if dimension == 2:
        # numbers = np.mgrid[0:1:_npj, 0:1:_npj]
        my_range = np.arange(10)/10

        foo = np.zeros(shape=(10, 2))
        foo[:, 0] = my_range

        bar = np.zeros(shape=(10, 2))
        bar[:, 0] = my_range
        bar[:, 1] = my_range
        return np.vstack((foo, bar))
    if dimension == 3:
        assert False, "Do something clever here"
        pass

    return Path("{:d}D_out_cressman".format(dim))
