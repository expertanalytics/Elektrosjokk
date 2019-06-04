import dolfin as df

from pathlib import Path
from typing import Union

from postfields import Field
from post import Saver
from coupled_brainmodel import CoupledBrainModel
from coupled_splittingsolver import BidomainSplittingSolver

from xalbrain.cellmodels import (
    Cressman,
    Noble
)

from coupled_utils import (
    get_mesh,
    CellTags,
    InterfaceTags,
    CoupledSplittingSolverParameters,
    CoupledODESolverParameters,
    CoupledBidomainParameters,
)

from postspec import (
    FieldSpec,
    SaverSpec,
)


def get_brain() -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh("new_meshes", "skullgmwm")
    cell_tags = CellTags(CSF=3, GM=2, WM=1)
    interface_tags = InterfaceTags(skull=3, CSF_GM=2, GM_WM=1, CSF=None, GM=None, WM=None)
    # mesh, cell_function, interface_function = get_mesh("meshes", "fine_all")
    # cell_tags = CellTags()
    # interface_tags = InterfaceTags()

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    Mi_dict = {
        1: df.Constant(1e-12),    # Set to zero?
        2: df.Constant(1),        # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        3: df.Constant(1),        # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
    }

    Me_dict = {
        1: df.Constant(16.54),     # Dougherty isotropic CSF conductivity 16.54 [mS/cm]
        2: df.Constant(2.76),      # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
        3: df.Constant(1.26),      # Dougherty isotropic "M extracellular conductivity 1.26 [mS/cm]
    }

    A = -800
    a = 0.01
    x0 = -46.4676
    y0 = 63.478
    expr_str = "A*exp(-a*(pow(x[0] - x0, 2) + pow(x[1] - y0, 2)))*abs(sin(t*2*pi*1e-3/20))"     # 20 Hz?
    applied_current = df.Expression(expr_str, degree=1, A=A, a=a, x0=x0, y0=y0, t=time_constant)

    neumann_bc_dict = {
        3: applied_current      # Applied to skull
    }

    brain = CoupledBrainModel(
        time=time_constant,
        mesh=mesh,
        cell_model=Cressman(),
        cell_function=cell_function,
        cell_tags=cell_tags,
        interface_function=interface_function,
        interface_tags=interface_tags,
        intracellular_conductivity=Mi_dict,
        other_conductivity=Me_dict,         # Either lmbda or extracellular
        neumann_boundary_condition=neumann_bc_dict,
        surface_to_volume_factor=Chi,
        membrane_capacitance=Cm
    )
    return brain


def get_solver(brain) -> BidomainSplittingSolver:
    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=(2,),
        reload_extension_modules=False
    )
    pde_parameters = CoupledBidomainParameters(linear_solver_type="direct")
    # pde_parameters = CoupledBidomainParameters()
    solver = BidomainSplittingSolver(
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

    field_spec_checkpoint = FieldSpec(save_as=("xdmf"), stride_timestep=40)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("xdmf"), stride_timestep=40*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))
    return saver


if __name__ == "__main__":
    brain = get_brain()
    solver = get_solver(brain)
    saver = get_saver(brain, "Test_bi")

    for i, solution_struct in enumerate(solver.solve(0, 1e3, 0.025)):
        print(f"{i} -- {brain.time(0)} -- {solution_struct.vur.vector().norm('l2')}")
        v, u, *_ = solution_struct.vur.split(deepcopy=True)
        update_dict = {
            "v": v,
            "u": u,
            "vs": solution_struct.vs,
        }
        saver.update(brain.time, i, update_dict)
    saver.close()