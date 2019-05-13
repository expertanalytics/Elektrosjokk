import dolfin as df

from pathlib import Path
from typing import Union

from postfields import Field
from post import Saver
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

from postspec import (
    FieldSpec,
    SaverSpec,
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

    vs_prev, *_ = solver.solution_fields()
    vs_prev.assign(brain.cell_model.initial_conditions())
    return solver


def get_saver(
        brain: CoupledBrainModel,
        outpath: Union[str, Path]
) -> Saver:
    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh)

    space_stride = 40
    field_spec = FieldSpec(
        save_as=("xdmf", "hdf5"),       # TODO: skip hdf5? -- If I can read from xdmf
        stride_timestep=space_stride
    )
    saver.add_field(Field("v", field_spec))
    saver.add_field(Field("vs", field_spec))
    return saver


if __name__ == "__main__":
    brain = get_brain()
    solver = get_solver()
    saver = get_saver(brain, "Test")

    for i, solution_struct in enumerate(solver.solve(0, 1, 1e-2)):
        print(f"{i} -- {solution_struct.vs.vector().norm('l2')}")
        update_dict = {
            "v": solution_struct.vur,
            "vs": solution_struct.vs,
        }
        saver.update(brain.time, i, update_dict)
    saver.close()

