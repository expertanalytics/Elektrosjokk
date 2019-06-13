import dolfin as df

from pathlib import Path
from typing import Union

from postfields import Field
from post import Saver
from coupled_brainmodel import CoupledBrainModel
from coupled_splittingsolver import (
    NetworkMonodomainSplittingSolver,
    MonodomainSplittingSolver,
)

from xalbrain.cellmodels import (
    Cressman,
    Noble
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
    CoupledSplittingSolverParameters,
    CoupledODESolverParameters,
    CoupledMonodomainParameters,
)

from postspec import (
    FieldSpec,
    SaverSpec,
)

from IPython import embed


def get_mesh():
    mesh = df.UnitIntervalMesh(2000)
    mesh.coordinates()[:] *= 10      # 10 cm

    CSF = df.CompiledSubDomain("x[0] < 1.5 || x[0] > 3.5")

    cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    cell_function.set_all(2)        # GM
    CSF.mark(cell_function, 3)      # CSF

    interface_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
    interface_function.set_all(0)

    return mesh, cell_function, interface_function


def get_brain() -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh()
    cell_tags = CellTags(CSF=2, GM=3, WM=3)
    interface_tags = InterfaceTags(skull=None, CSF_GM=None, GM_WM=None, CSF=None, GM=None, WM=None)

    Mi = 1
    lgm = Mi/2.76
    lwm = Mi/1.26

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    # Include Cm and Chi as well
    Mi_dict = {
        # 1: df.Constant(16.54),              # Dlougherty isotropic CSF conductivity 16.54 [mS/cm]
        2: df.Constant(Mi/(Chi*Cm))*df.Constant(lgm/(1 + lgm)),        # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        3: df.Constant(Mi/(Chi*Cm))*df.Constant(lgm/(1 + lgm)),        # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
    }

    lambda_dict = {
        # 1: df.Constant(1),                  # I don't know what to do here yet
        2: df.Constant(lgm/(1 + lgm)),
        3: df.Constant(lgm/(1 + lgm)),      # lwm
    }

    # external_stimulus = {2: df.Constant(-1.0)}

    # A = 200
    # a = 0.1
    # x0 = -46.4676
    # y0 = 63.478
    # expr_str = "A*exp(-a*(pow(x[0] - x0, 2) + pow(x[1] - y0, 2)))*sin(t*2*pi*1e-3/20)"     # 2 Hz?
    # applied_current = df.Expression(expr_str, degree=1, A=A, a=a, x0=x0, y0=y0, t=time_constant)

    # neumann_bc_dict = {
    #     6: applied_current
    # }

    kinf_str = "x[0] < {d1} || x[0] > {d2} ? {K1} : {K2}"
    Kinf = df.Expression(kinf_str.format(d1=1.5, d2=3.5, K1=4, K2=8), degree=1)
    cressman_parameters = Cressman.default_parameters()
    cressman_parameters["Koinf"] = Kinf
    cell_model = Cressman(params=cressman_parameters)

    brain = CoupledBrainModel(
        time=time_constant,
        mesh=mesh,
        cell_model=cell_model,
        cell_function=cell_function,
        cell_tags=cell_tags,
        interface_function=interface_function,
        interface_tags=interface_tags,
        intracellular_conductivity=Mi_dict,
        other_conductivity=lambda_dict,
        # neumann_boundary_condition=neumann_bc_dict
        # external_stimulus=external_stimulus
    )
    return brain


def get_solver(brain, timestep) -> Union[NetworkMonodomainSplittingSolver, MonodomainSplittingSolver]:
    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=[2, 3],
        reload_extension_modules=False
    )
    pde_parameters = CoupledMonodomainParameters(
        theta=0.5,
        timestep=timestep,
        linear_solver_type="direct",
        krylov_method="cg",
        krylov_preconditioner="petsc_amg"
    )
    solver = NetworkMonodomainSplittingSolver(      # Working
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )

    vs_prev, *_ = solver.solution_fields()

    default_ic = brain.cell_model.default_initial_conditions()
    stable_ic = [
        -5.88804333e+01,
        3.31806918e-02,
        1.30223180e-01,
        9.29826068e-01,
        3.71094385e-06,
        6.96598540e+00,
        1.67147481e+01
    ]

    for key, new_value in zip(default_ic, stable_ic):
        default_ic[key] = new_value

    brain.cell_model.set_initial_conditions(**default_ic)
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

    field_spec_checkpoint = FieldSpec(save_as=("xdmf"), stride_timestep=40*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))
    return saver


if __name__ == "__main__":
    brain = get_brain()
    timestep = 0.025
    solver = get_solver(brain, timestep)
    saver = get_saver(brain, "Test_mono")

    _, vs, vur = solver.solution_fields()
    update_dict = {"v": vur, "vs": vs}
    # saver.store_initial_condition(update_dict)  # v_0 is not set. Updateds internally

    for i, solution_struct in enumerate(solver.solve(0, 1e3, timestep)):
        print(f"{i} -- {brain.time(0)} -- {solution_struct.vur.vector().norm('l2')}")
        update_dict["v"] = solution_struct.vur
        update_dict["vs"] = solution_struct.vs
        saver.update(brain.time, i + 1, update_dict)
    saver.close()
