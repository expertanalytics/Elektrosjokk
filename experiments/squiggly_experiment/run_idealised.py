import time
import resource
import math
import warnings
import datetime
import sys
import itertools

import dolfin as df
import numpy as np

from pathlib import Path
from scipy import signal
from math import pi
from multiprocessing import Pool

from poisson import get_anisotropy

from postfields import (
    Field,
    PointField,
)

from typing import (
    Union,
    Dict,
    Sequence,
    Tuple,
)

from extension_modules import load_module

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
    Noble,
    MorrisLecar,
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
    CoupledSplittingSolverParameters,
    CoupledODESolverParameters,
    CoupledBidomainParameters,
    get_mesh,
)

from postspec import (
    FieldSpec,
    SaverSpec,
    LoaderSpec,
)


def get_brain(case_id, conductivity) -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh("squiggly_meshes", "squiggly")

    cell_tags = CellTags(CSF=3, GM=2, WM=1, Kinf=4)
    interface_tags = InterfaceTags(skull=None, CSF_GM=None, GM_WM=None, CSF=None, GM=None, WM=None)

    test_cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    test_cell_function.set_all(0)

    df.CompiledSubDomain("x[1] >= -pi/16*x[0]").mark(test_cell_function, 4)
    if case_id == 1:
        df.CompiledSubDomain("x[1] >= 25*pi/16*x[0]").mark(test_cell_function, 2)
    elif case_id == 2:
        df.CompiledSubDomain("x[1] >= 7*pi/32*x[0]").mark(test_cell_function, 2)

    cell_function.array()[(cell_function.array() == 2) & (test_cell_function.array() == 4)] = 4

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    loader = Loader(LoaderSpec(casedir="anisotropy"))
    _, ngp = next(loader.load_field("normalised_grad_poisson", vector=True))
    white_Mi = get_anisotropy(ngp, 0.1, 0.01)
    white_Me = get_anisotropy(ngp, 0.276, 0.0276)

    Mi_dict = {
        3: df.Constant(1e-12),       # Set to zero?    1e-12
        1: df.Constant(conductivity),           # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
        2: df.Constant(conductivity),           # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        4: df.Constant(conductivity),           # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
    }

    Me_dict = {
        3: df.Constant(16.54),        # Dougherty isotropic CSF conductivity 16.54 [mS/cm] 16.54
        1: df.Constant(1.26),         # 1.26     # Dougherty isotropic WM extracellular conductivity 1.26 [mS/cm]
        2: df.Constant(2.76),         # 2.76     # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
        4: df.Constant(2.76),         # 2.76     # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
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


def get_solver(brain, Ks, Ku) -> BidomainSplittingSolver:

    odesolver_module = load_module("LatticeODESolver")
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(1, odesolver_module.MorrisLecar())
    odemap.add_ode(2, odesolver_module.Cressman(Ks))
    odemap.add_ode(4, odesolver_module.Cressman(Ku))

    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=(1, 2, 4),
        reload_extension_modules=False,
        parameter_map=odemap
    )

    pde_parameters = CoupledBidomainParameters(linear_solver_type="direct")

    solver = BidomainSplittingSolver(
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )

    vs_prev, *_ = solver.solution_fields()

    cressman_values = list(Cressman.default_initial_conditions().values())
    morris_lecar_values = list(FitzHughNagumoManual.default_initial_conditions().values())

    morris_lecar_full_values = [0]*len(cressman_values)
    morris_lecar_full_values[:len(morris_lecar_values)] = morris_lecar_values

    csf_values = [0]*len(cressman_values)

    _, cell_function, _ = get_mesh("squiggly_meshes", "squiggly")

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
    outpath: Union[str, Path],
    case_index: int
) -> Saver:
    sourcefiles = [
        "coupled_bidomain.py",
        "coupled_brainmodel.py",
        "coupled_odesolver.py",
        "coupled_splittingsolver.py",
        "coupled_utils.py",
        "run_idealised.py",
    ]
    store_sourcefiles(map(Path, sourcefiles), outpath)

    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh)

    field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=20)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("hdf5"), stride_timestep=20*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))

    trace_names = []
    r = np.linspace(-9, 9, )
    points = np.zeros(shape=(r.size, 2))
    for sub_index in range(7):
        point_field_spec = FieldSpec(stride_timestep=20, sub_field_index=sub_index)
        for theta in [1, 5, 9, 13]:
            points[:, 0] = r*np.cos(theta)
            points[:, 1] = r*np.sin(theta)
            name = "trace{}_{}".format(sub_index, theta)
            saver.add_field(PointField(name, point_field_spec, points))
            trace_names.append(name)

            angular_offset = pi/8
            points[:, 0] = r*np.cos(theta + angular_offset)
            points[:, 1] = r*np.sin(theta + angular_offset)
            name = "trace_offset{}_{}".format(sub_index, theta)
            saver.add_field(PointField(name, point_field_spec, points))
            trace_names.append(name)

    return saver, trace_names


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)


    def run(args):
        conductivity, case_id, Ks, Ku = args
        T = 1e1
        dt = 0.05

        brain = get_brain(case_id, conductivity)
        solver = get_solver(brain, Ks, Ku)
        print("got solver")

        identifier = simulation_directory(
            home=Path("."),
            parameters={
                 "time": datetime.datetime.now(),
                 "case_id": case_id,
                 "conductivity": conductivity,
                 "Ks": Ks,
                 "Ku": Ku
            },
            directory_name="squiggly"
        )

        saver, trace_name_list = get_saver(brain, identifier, case_id)

        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        tick = time.perf_counter()

        tick = time.perf_counter()
        for i, solution_struct in enumerate(solver.solve(0, T, dt)):
            tock = time.perf_counter()
            print(f"time: {tock - tick}")
            norm = solution_struct.vur.vector().norm('l2')
            if math.isnan(norm):
                print("nanananananana --> BREAKING!")
                break
            print(f"{i} -- {brain.time(0):.5f} -- {norm:.6e}")

            update_dict = dict()
            if saver.update_this_timestep(field_names=["u", "v"], timestep=i, time=brain.time(0)):
                v, u, *_ = solution_struct.vur.split(deepcopy=True)
                update_dict.update({"v": v, "u": u})

            if saver.update_this_timestep(field_names=trace_name_list, timestep=i, time=brain.time(0)):
                update_dict.update({k: solution_struct.vs for k in trace_name_list})

            if saver.update_this_timestep(field_names=["vs"], timestep=i, time=brain.time(0)):
                update_dict.update({"vs": solution_struct.vs})

            if len(update_dict) != 0:
                saver.update(brain.time, i, update_dict)
            tick = tock

        saver.close()
        tock = time.perf_counter()
        max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
        print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
        print("Execution time: {:.2f} s".format(tock - tick))

    run((1, 1, 4, 8))

    # conductivities = [2**(2*n) for n in range(-3, 2)]
    # lengths = list(range(3))

    # conductivities = [1]
    # lengths = [1]

    # Ks = float(sys.argv[1])
    # Ku = float(sys.argv[2])

    # parameter_list = list(itertools.product(conductivities, lengths, [Ks], [Ku]))
    # parameter_list = parameter_list[:2]

    # pool = Pool(processes=len(parameter_list))
    # pool.map(run, parameter_list)
