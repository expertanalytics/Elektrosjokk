import logging

import xalbrain as xb
import numpy as np

from ect.specs import (
    SolutionFieldSpec,
    PDESolverSpec,
    PDESimulationSpec,
)

from ect.utilities import (
    NonuniformIC,
    project_ic,
)

from typing import (
    Dict,
    Tuple,
)

from cbcpost import (
    PostProcessor,
    SolutionField
)


logger = logging.getLogger(name=__name__)


def solve_pde(
        brain: xb.CardiacModel, 
        pde_parameters: PDESolverSpec,
        simulation_pec: PDESimulationSpec,
        field_spec: SolutionFieldSpec,
        uniform_ic: Dict[str, float] = None,
        nonuniform_ic_generator: NonuniformIC = None,
        outdir: str = None
) -> Tuple[float, Tuple[xb.Function]]:
    """Solve the bidomain model coupled with an ode."""
    # FIXME: This is ugly
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
    vs_, vs, vur = solver.solution_fields()

    if uniform_ic is not None:
        brain.cell_models().set_initial_conditions(**uniform_ic)
        vs_.assign(brain.cell_models().initial_conditions())
    elif nonuniform_ic_generator is not None:
        project_ic(vs_, nonuniform_ic_generator)
    else:
        # Use default ICs
        vs_.assign(brain.cell_models().initial_conditions())        # Use defaults

    # Extract end time and time-step from application parameters
    end_time = simulation_pec.end_time
    dt = simulation_pec.timestep

    postprocessor = None
    if outdir is not None:
        postprocessor = PostProcessor(dict(casedir=outdir, clean_casedir=True))
        postprocessor.store_mesh(brain.mesh)
        postprocessor.add_field(SolutionField("v", field_spec._asdict()))
        postprocessor.add_field(SolutionField("u", field_spec._asdict()))

    theta = pde_parameters.theta
    # Solve forward problem
    N = end_time / dt + 1
    for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, end_time), dt)):
        functions = vs.split()
        current_t = t0 + theta*(t1 - t0)

        if postprocessor is not None:
            v, u = vur.split(deepcopy=True)
            postprocessor.update_all({"v": lambda: v, "u": lambda: u}, current_t, i)
            # postprocessor.update_all({"vs": lambda: vs}, current_t, i)

        logger.debug(f"step: {i}, norm: {vur.vector().norm('l2')}")
        logger.info("timestep {:>10}/{:>10}".format(i, N))
        yield current_t, functions

    if postprocessor is not None:
        postprocessor.finalize_all()
