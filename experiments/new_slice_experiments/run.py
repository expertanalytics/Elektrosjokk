import dolfin as df

from typing import (
    Tuple,
)

from coupled_brainmodel import CoupledBrainModel

from coupled_splittingsolver import CoupledSplittingsolver

from xalbrain.cellmodels import (
    Cressman,
    Noble
)

from coupled_utils import (
    get_mesh,
    CellTags,
    InterfaceTags,
    CoupledSplittingsolverParameters,
    CoupledODESolverParameters,
    CoupledMonodomainParameters,
)


def get_brain() -> CoupledBrainModel:
    mesh, cell_function, interface_function = get_mesh("meshes", "fine_all")
    cell_tags = CellTags()
    interface_tags = InterfaceTags()

    Mi = 1
    lgm = Mi/2.76
    lwm = Mi/1.26

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    # Include Cm and Chi as well
    Mi_dict = {
        1: df.Constant(16.54),                   # Dlougherty isotropic CSF conductivity 16.54 [mS/cm]
        2: df.Constant(Mi),        # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        3: df.Constant(Mi),        # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
    }

    lambda_dict = {
        1: df.Constant(1),           # I don't know what to do here yet
        2: df.Constant(lgm),
        3: df.Constant(lwm),
    }

    time_constant = df.Constant(0)
    brain = CoupledBrainModel(
        time=time_constant,
        mesh=mesh,
        cell_model=Cressman(),
        cell_function=cell_function,
        cell_tags=cell_tags,
        interface_function=interface_function,
        interface_tags=interface_tags,
        intracellular_conductivity=Mi_dict,
        other_conductivity=lambda_dict
    )
    return brain


def get_solver() -> CoupledSplittingsolver:
    brain = get_brain()
    parameters = CoupledSplittingsolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=[2],
        reload_extension_modules=False
    )
    pde_parameters = CoupledMonodomainParameters()
    solver = CoupledSplittingsolver(
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )
    return solver


if __name__ == "__main__":
    # get_brain()
    solver = get_solver()
    for i, _ in enumerate(solver.solve(0, 1, 1e-2)):
        print(i)
