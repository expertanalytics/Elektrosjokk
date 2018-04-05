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
    NonuniformIC,
)



def make_brain():
    """Create a CardiacModel for storing relevant parameters."""
    time = xb.Constant(0)

    N = 2
    mesh = xb.UnitSquareMesh(N, N)
    M_i, M_e = setup_conductivities(mesh)
    cell_model = xb.cellmodels.Wei()
    args = (mesh, time, M_i, M_e, cell_model)
    return xb.CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus=None)


def main():
    brain = make_brain()
    solver_spec = PDESolverSpec(theta=0.5, ode_scheme="RK4")
    field_spec = SolutionFieldSpec()
    simulation_spec = PDESimulationSpec(end_time=2e1, timestep=1e-2)

    REFERENCE_SOLUTION = np.load("REFERENCE_SOLUTION.npy")
    # fire_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="fire")
    model = brain.cell_models() 

    _start = 192300
    coordinates = brain.mesh.coordinates()
    ic_generator = NonuniformIC(
        coordinates[:, 0],
        REFERENCE_SOLUTION[_start:_start + 300]
    )

    solution_generator = solve_pde(
        brain,
        solver_spec,
        simulation_spec,
        field_spec,
        uniform_ic=None,
        nonuniform_ic_generator=ic_generator,
        outdir="test"
    )

    dataframe = create_dataframe(
        solution_generator,
        model.default_initial_conditions().keys(),      # field names
        (0.5, 0.5),
    )
    dataframe.to_pickle("theta1.pkl")


if __name__ == "__main__":
    main()
