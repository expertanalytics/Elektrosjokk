import time
import resource
import warnings
import datetime

import dolfin as df
import numpy as np

from pathlib import Path
from scipy import signal
from math import pi

from postfields import (
    Field,
    PointField,
)

from typing import (
    Union,
    Tuple,
)


from post import Saver
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
    Noble
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
)


def compute_initial_condirions(brain, vs_prev, zero_tags):
    mesh = brain._mesh
    cell_function = brain._cell_function

    dX = df.Measure("dx", domain=mesh, subdomain_data=cell_function)

    V = df.FunctionSpace(mesh, "DG", 0)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    sol = df.Function(V)

    value = brain.cell_model.default_initial_conditions()["V"]

    F = 0
    for tag in set(brain.cell_tags) - set(zero_tags):
        F += -u*v*dX(tag) + df.Constant(value)*v*dX(tag)

    a = df.lhs(F)
    L = df.rhs(F)

    A = df.assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = df.assemble(L)

    solver = df.KrylovSolver("cg", "petsc_amg")
    solver.set_operator(A)
    solver.solve(sol.vector(), b)

    VCG = df.FunctionSpace(mesh, "CG", 1)
    v_new = df.Function(VCG)
    v_new.interpolate(sol)

    Vp = vs_prev.function_space().sub(0)
    merger = df.FunctionAssigner(Vp, VCG)
    merger.assign(vs_prev.sub(0), v_new)


def get_brain(i) -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh("idealised_meshes", "idealised{i}".format(i=i))

    test_cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    test_cell_function.set_all(0)
    df.CompiledSubDomain("x[0] > -0.33 && x[0] < 0.33").mark(test_cell_function, 4)

    # Hack?
    cell_function.array()[(cell_function.array() == 2) & (test_cell_function.array() == 4)] = 4

    cell_tags = CellTags(CSF=3, GM=2, WM=1, Kinf=4)
    interface_tags = InterfaceTags(skull=None, CSF_GM=None, GM_WM=None, CSF=None, GM=None, WM=None)

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    Mi_dict = {
        3: df.Constant(1e-12),    # Set to zero?
        1: df.Constant(1),        # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        2: df.Constant(1),        # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
        4: df.Constant(1),       # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
    }

    Me_dict = {
        3: df.Constant(16.54),     # Dougherty isotropic CSF conductivity 16.54 [mS/cm]
        1: df.Constant(1.26),      # Dougherty isotropic WM extracellular conductivity 1.26 [mS/cm]
        2: df.Constant(2.76),      # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
        4: df.Constant(2.76),      # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
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
    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=(2, 4),
        reload_extension_modules=False
    )

    pde_parameters = CoupledBidomainParameters(
        linear_solver_type="direct"
    )

    solver = BidomainSplittingSolver(
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )

    vs_prev, *_ = solver.solution_fields()
    vs_prev.assign(brain.cell_model.initial_conditions())
    compute_initial_condirions(brain, vs_prev, {3})
    return solver


def get_saver(
        brain: CoupledBrainModel,
        outpath: Union[str, Path]
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

    field_spec_checkpoint = FieldSpec(save_as=("xdmf"), stride_timestep=40)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("xdmf"), stride_timestep=40*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))

    point_list = (
        # K8
        (0.5, 0.5),
        (0.6, 0.6),
        (0.4, 0.6),
        # K8
        (0.5, 0.3),
        (0.5, 0.9),
        # WM
        (0.2, 0.5),
        # CSF
        (0.8, 0.5)
    )

    point_field_spec = FieldSpec(stride_timestep=1)
    for i, centre in enumerate(point_list):
        points = circle_points(radii=[0, 0.1, 0.2, 0.3], num_points=[1, 6, 18, 24], r0=centre)
        saver.add_field(PointField("psd_v{}".format(i), point_field_spec, points))
    return saver


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    from multiprocessing import Pool

    def run(i):
        N = 300
        T = 1e1
        dt = 0.025

        brain = get_brain(i)
        solver = get_solver(brain)

        identifier = simulation_directory(
            parameters={"time": datetime.datetime.now(), "i": i},
            directory_name=".simulations/idealised"
        )

        saver = get_saver(brain, identifier)

        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        tick = time.perf_counter()
        for i, solution_struct in enumerate(solver.solve(0, T, dt)):
            print(f"{i} -- {brain.time(0)} -- {solution_struct.vur.vector().norm('l2')}")
            v, u, *_ = solution_struct.vur.split(deepcopy=True)
            update_dict = {
                "v": v,
                "u": u,
                "psd_v0": v,
                "psd_v1": v,
                "psd_v2": v,
                "psd_v3": v,
                "psd_v4": v,
                "psd_v5": v,
                "psd_v6": v,
            }
            saver.update(brain.time, i, update_dict)
        saver.close()
        tock = time.perf_counter()
        max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
        print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
        print("Execution time: {:.2f} s".format(tock - tick))

    parameter_list = [1, 2, 3, 4, 5, 6, 7, 8]
    pool = Pool(processes=len(parameter_list))
    pool.map(run, parameter_list)
