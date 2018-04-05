import logging

import xalbrain as xb
import numpy as np

from ect.specs import (
    SolutionFieldSpec,
    PDESolverSpec,
    PDESimulationSpec,
)

from typing import (
    Dict,
    Tuple,
)


logger = logging.getLogger(name=__name__)


def solve_pde(
        brain: xb.CardiacModel, 
        pde_parameters: PDESolverSpec,
        simulation_pec: PDESimulationSpec,
        field_spec: SolutionFieldSpec,
        initial_conditions: Dict[str, float] = None,
        verbose: bool = True
) -> Tuple[float, Tuple[xb.Function]]:
    """Solve the bidomain model coupled with an ode."""

    # Customize and create a splitting solver
    splitting_params = xb.SplittingSolver.default_parameters()
    splitting_params["pde_solver"] = "bidomain"
    splitting_params["theta"] = pde_parameters.theta
    splitting_params["CardiacODESolver"]["scheme"] = pde_parameters.ode_scheme
    splitting_params["BidomainSolver"]["linear_solver_type"] = pde_parameters.linear_solver_type
    splitting_params["BidomainSolver"]["algorithm"] = \
        pde_parameters.linear_solver
    splitting_params["BidomainSolver"]["preconditioner"] = \
        pde_parameters.preconditioner
    splitting_params["BidomainSolver"]["use_avg_u_constraint"] = \
        pde_parameters.avg_u_constraint
    splitting_params["apply_stimulus_current_to_pde"] = pde_parameters.pde_stimulus
    splitting_params["BidomainSolver"]["petsc_krylov_solver"]["absolute_tolerance"] = \
        pde_parameters.krylov_absolute_tolerance
    splitting_params["BidomainSolver"]["petsc_krylov_solver"]["relative_tolerance"] = \
        pde_parameters.krylov_relative_tolarance
    splitting_params["BidomainSolver"]["petsc_krylov_solver"]["nonzero_initial_guess"] = \
        pde_parameters.krylov_nonzero_initial_guess

    # TODO: Splittingsolver should use namedtuples
    solver = xb.SplittingSolver(brain, params=splitting_params)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    # vs_.assign(brain.cell_models().initial_conditions())

    if initial_conditions is not None:
        brain.cell_models().set_initial_conditions(**initial_conditions)
    vs_.assign(brain.cell_models().initial_conditions())        # Use defaults
    # assign_ic(vs_)        # TODO: this need another interface

    # Extract end time and time-step from application parameters
    end_time = simulation_pec.end_time
    dt = simulation_pec.timestep

    # postprocessor = PostProcessor(dict(casedir="test", clean_casedir=True))
    # postprocessor.store_mesh(brain.domain())

    # postprocessor.add_field(SolutionField("v", SolutionFieldSpec._asdict()))
    # postprocessor.add_field(SolutionField("u", SolutionFieldSpec._asdict()))
    # postprocessor.add_field(SolutionField("s", SolutionFieldSpec._asdict()))

    theta = pde_parameters.theta
    # Solve forward problem
    N = end_time / dt + 1
    for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, end_time), dt)):
        functions = vs.split()
        current_t = t0 + theta*(t1 - t0)

        # v, u = vur.split(deepcopy=True)
        # v = vur.split(deepcopy=True)

        logger.debug(f"step: {i}, norm: {vur.vector().norm('l2')}")
        if i % int(field_spec.stride_timestep) == 0:
            if verbose:
                print("timestep {:<10}/{:>10}".format(i, N))
            logger.info("timestep {:>10}/{:>10}".format(i, N))
            # postprocessor.update_all({"v": lambda: v, "u": lambda: u}, current_t, i)
            # postprocessor.update_all({"vur": lambda: vur}, current_t, i)
            # postprocessor.update_all({"vs": lambda: vs}, current_t, i)
            yield current_t, functions

    # postprocessor.finalize_all()
