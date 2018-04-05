from typing import (
    NamedTuple,
    Tuple,
)


class PDESolverSpec(NamedTuple):
    theta: float = 0.5
    ode_scheme: str = "RK4"
    linear_solver_type: str = "iterative"
    linear_solver: str = "gmres"
    preconditioner: str = "petsc_amg"
    avg_u_constraint: bool = False
    pde_stimulus: bool = False
    krylov_absolute_tolerance: float = 1e-15
    krylov_relative_tolarance: float = 1e-15
    krylov_nonzero_initial_guess: bool = False


class SolutionFieldSpec(NamedTuple):
    save: bool = True
    save_as: Tuple[str] = ("hdf5", "xdmf")
    plot: bool = False
    start_timestep: int = -1
    stride_timestep: int = 1


class PDESimulationSpec(NamedTuple):
    end_time: float 
    timestep: float
    solution_direcctory: str = "results"
