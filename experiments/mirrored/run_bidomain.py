import time
import datetime
import math

import dolfin as df
import numpy as np

from pathlib import Path
from scipy import signal
from math import pi

from poisson import get_anisotropy

from typing import (
    Union,
    Sequence,
    Dict,
)

from postfields import (
    Field,
    PointField,
)

from post import Saver, Loader
from coupled_brainmodel import CoupledBrainModel
from coupled_splittingsolver import BidomainSplittingSolver

from postutils import (
    interpolate_ic,
    store_sourcefiles,
    simulation_directory,
    circle_points,
)

from xalbrain.cellmodels import (
    Cressman,
    FitzHughNagumoManual,
    MorrisLecar,
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
    LoaderSpec,
)

from extension_modules import load_module


def get_brain(i: int) -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh("mirrored_meshes", "mirrored")

    test_cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    test_cell_function.set_all(0)

    if i == 0:
        df.CompiledSubDomain("x[1] > -0.2769*x[0] + 4.34064").mark(test_cell_function, 4)
        df.CompiledSubDomain("x[1] > -3.8399*x[0] + -0.24458").mark(test_cell_function, 2)
    elif i == 1:
        df.CompiledSubDomain("x[1] > -0.2769*x[0] + 4.34064").mark(test_cell_function, 4)
        df.CompiledSubDomain("x[1] > -1.3659*x[0] + 2.56003").mark(test_cell_function, 2)
    elif i == 2:
        df.CompiledSubDomain("x[1] > -0.2769*x[0] + 4.34064").mark(test_cell_function, 4)
        df.CompiledSubDomain("x[1] > -0.4743*x[0] + 4.57442").mark(test_cell_function, 2)

    cell_function.array()[(cell_function.array() == 2) & (test_cell_function.array() == 4)] = 4

    cell_tags = CellTags(CSF=3, GM=2, WM=1, Kinf=4)
    interface_tags = InterfaceTags(skull=3, CSF_GM=2, GM_WM=1, CSF=None, GM=None, WM=None)

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    Mi_dict = {
        3: df.Constant(1e-12),    # Set to zero?
        2: df.Constant(1),        # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        4: df.Constant(1),       # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
        1: df.Constant(1),
    }

    Me_dict = {
        3: df.Constant(16.54),     # Dougherty isotropic CSF conductivity 16.54 [mS/cm]
        2: df.Constant(2.76),      # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
        4: df.Constant(2.76),      # Dougherty isotropic "M extracellular conductivity 1.26 [mS/cm]
        1: df.Constant(1.26),
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
        surface_to_volume_factor=Chi,
        membrane_capacitance=Cm
    )
    return brain


def get_solver(brain) -> BidomainSplittingSolver:

    odesolver_module = load_module("LatticeODESolver")
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(1, odesolver_module.MorrisLecar())
    odemap.add_ode(2, odesolver_module.Cressman())
    odemap.add_ode(4, odesolver_module.Cressman())

    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=(1, 2, 4),
        reload_extension_modules=False,
        parameter_map=odemap
    )

    pde_parameters = CoupledBidomainParameters(linear_solver_type="iterative")

    solver = BidomainSplittingSolver(
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )

    vs_prev, *_ = solver.solution_fields()

    cressman_values = list(Cressman.default_initial_conditions().values())
    morris_lecar_values = list(MorrisLecar.default_initial_conditions().values())

    morris_lecar_full_values = [0]*len(cressman_values)
    morris_lecar_full_values[:len(morris_lecar_values)] = morris_lecar_values

    csf_values = [0]*len(cressman_values)

    _, cell_function, _ = get_mesh("mirrored_meshes", "mirrored")
    cell_model_dict = {
        1: morris_lecar_full_values,
        3: csf_values,
        2: cressman_values,
    }

    odesolver_module.assign_vector(
        vs_prev.vector(),
        cell_model_dict,
        cell_function,
        vs_prev.function_space()._cpp_object
    )

    return solver


def get_saver(
        brain: CoupledBrainModel,
        outpath: Union[str, Path]
) -> Saver:
    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh)

    sourcefiles = [
        "coupled_bidomain.py",
        "coupled_brainmodel.py",
        "coupled_odesolver.py",
        "coupled_splittingsolver.py",
        "coupled_utils.py",
        "run_bidomain.py"
    ]

    store_sourcefiles(map(Path, sourcefiles), outpath)

    field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=20*10)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("hdf5"), stride_timestep=20*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))

    point_list = (
        # K8
        (-38.95, 59.43),
        (-35.75, 67.66),
        (-26.97, 75.30),
        # K4
        (-16.59, 78.77),
        (-48.33, 48.47),
        (-27.66, 38.53)
    )

    edge_points = (
       (-52.34, 46.96),
       (-48.38, 56.68),
       (-44.91, 62.43),
       (-39.65, 70.66),
       (-35.89, 73.83),
       (-30.43, 79.78),
       (-22.40, 82.73),
       (-16.35, 84.89)
    )

    point_field_spec = FieldSpec(stride_timestep=20, sub_field_index=0)
    for i, centre in enumerate(point_list):
        points = circle_points(radii=[0, 0.1, 0.2, 0.3], num_points=[1, 6, 18, 24], r0=centre)
        saver.add_field(PointField("trace_v{}".format(i), point_field_spec, points))

    point_field_spec = FieldSpec(stride_timestep=20, sub_field_index=1)
    for i, centre in enumerate(point_list):
        points = circle_points(radii=[0, 0.1, 0.2, 0.3], num_points=[1, 6, 18, 24], r0=centre)
        saver.add_field(PointField("trace_u{}".format(i), point_field_spec, points))

    for i, centre in enumerate(edge_points):
        points = circle_points(radii=[0, 0.1], num_points=[1, 6], r0=centre)
        saver.add_field(PointField("trace_csf{}".format(i), point_field_spec, points))

    point = (-37, 58)
    for i in range(7):
        point_field_spec = FieldSpec(stride_timestep=20, sub_field_index=i)
        saver.add_field(PointField(f"trace_state{i}", point_field_spec, point))

    return saver


if __name__ == "__main__":
    brain = get_brain(1)
    solver = get_solver(brain)

    identifier = simulation_directory(
        parameters={"time": datetime.datetime.now()},
        directory_name=Path("results"),
        home="."
    )

    saver = get_saver(brain, identifier)
    traces = [f"trace_v{i}" for i in range(6)]
    traces += [f"trace_u{i}" for i in range(6)]
    traces += [f"trace_csf{i}" for i in range(8)]
    state_traces = [f"trace_state{i}" for i in range(7)]

    tick = time.perf_counter()
    for i, solution_struct in enumerate(solver.solve(0, 1e2, 0.05)):
        tock = time.perf_counter()
        print(f"time: {tock - tick}")
        norm = solution_struct.vur.vector().norm('l2')
        if math.isnan(norm):
            print(f"norm: {norm} -- BREAKING")
            break
        print(f"{i} -- {brain.time(0):.5f} -- {norm:.6e}")

        update_dict = dict()
        if saver.update_this_timestep(field_names=traces + state_traces, timestep=i, time=brain.time(0)):
            update_dict = {k: solution_struct.vur for k in traces}
            update_dict = {k: solution_struct.vs for k in state_traces}

        if saver.update_this_timestep(field_names=["v", "u"], timestep=i, time=brain.time(0)):
            v, u, *_ = solution_struct.vur.split(deepcopy=True)
            update_dict.update({"v": v, "u": u})

        if saver.update_this_timestep(field_names=["vs"], timestep=i, time=brain.time(0)):
            update_dict.update({"vs": solution_struct.vs})

        if len(update_dict) != 0:
            saver.update(brain.time, i, update_dict)
        tick = tock

    saver.close()
