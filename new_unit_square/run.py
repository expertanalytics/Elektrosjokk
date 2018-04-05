import logging

import xalbrain as xb
import numpy as np

from ect.pdesolvers import (
    solve_pde,
    setup_conductivities,
)

from ect.specs import (
    PDESolverSpec,
    PDESimulationSpec,
    SolutionFieldSpec,
)

from ect.utilities import (
    wei_uniform_ic,
    create_dataframe,
)


def make_brain():
    """Create a CardiacModel for storing relevant parameters."""
    time = xb.Constant(0)

    N = 2
    mesh = xb.UnitSquareMesh(N, N)
    M_i, M_e = setup_conductivities(mesh)     # The keys match cell_domains.array()
    cell_model = xb.cellmodels.Wei()
    # cell_model = xb.cellmodels.LogisticCellModel()
    # cell_model = xb.cellmodels.TestCellModel()

    # stimulus =  xb.Expression("10*(t < 25)", degree=0, t=time)
    stimulus = None
    args = (mesh, time, M_i, M_e, cell_model)
    return xb.CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus=stimulus)


def main():
    brain = make_brain()
    solver_spec = PDESolverSpec(theta=1.0, ode_scheme="RK4")
    field_spec = SolutionFieldSpec()
    simulation_spec = PDESimulationSpec(end_time=2e1, timestep=1e-2)

    REFERENCE_SOLUTION = np.load("REFERENCE_SOLUTION.npy")
    fire_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="fire")
    model = brain.cell_models() 

    solution_generator = solve_pde(
        brain,
        solver_spec,
        simulation_spec,
        field_spec,
        # initial_conditions=None,
        initial_conditions=fire_ic,
        verbose=True
    )

    dataframe = create_dataframe(
        solution_generator,
        model.default_initial_conditions().keys(),      # field names
        (0.5, 0.5),
    )
    dataframe.to_pickle("theta1.pkl")


if __name__ == "__main__":
    main()
