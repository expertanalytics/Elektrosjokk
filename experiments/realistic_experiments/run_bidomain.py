import time
import datetime

import dolfin as df
import numpy as np

from pathlib import Path
from typing import Union
from scipy import signal
from math import pi

from postfields import (
    Field,
    PointField,
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

from extension_modules import load_module


def load_array(name: str, directory: Union[Path, str] = "../data") -> np.ndarray:
    return np.loadtxt(str(Path(directory) / name), delimiter=",")


def get_brain() -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh("realistic_meshes", "skullgmwm_fine1")

    test_cell_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    test_cell_function.set_all(0)
    df.CompiledSubDomain("x[0] < -20 && x[1] > 55").mark(test_cell_function, 4)

    # Hack?
    cell_function.array()[(cell_function.array() == 2) & (test_cell_function.array() == 4)] = 4

    cell_tags = CellTags(CSF=3, GM=2, WM=1, Kinf=4)
    interface_tags = InterfaceTags(skull=3, CSF_GM=2, GM_WM=1, CSF=None, GM=None, WM=None)

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    Mi_dict = {
        1: df.Constant(1e-12),    # Set to zero?
        2: df.Constant(1),        # Dlougherty isotropic GM intracellular conductivity 1.0 [mS/cm]
        3: df.Constant(1),        # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
        4: df.Constant(1),       # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
    }

    Me_dict = {
        1: df.Constant(16.54),     # Dougherty isotropic CSF conductivity 16.54 [mS/cm]
        2: df.Constant(2.76e1),      # Dougherty isotropic GM extracellular conductivity 2.76 [mS/cm]
        3: df.Constant(1.26e1),      # Dougherty isotropic "M extracellular conductivity 1.26 [mS/cm]
        4: df.Constant(2.76e1),      # Dougherty isotropic "M extracellular conductivity 1.26 [mS/cm]
    }

    class Source(df.UserExpression):
        def __init__(self, frequency, amplitude, x0, alpha, time, **kwargs):
            super().__init__(kwargs)
            self._frequency = frequency     # in Hz
            self._amplitude = amplitude
            self._x0, self._y0 = x0
            self._alpha = alpha
            self._time = float(time)

        def eval(self, values, x):
            t = self._time
            A = self._amplitude*signal.square(t*2*pi*self._frequency)
            x0 = self._x0
            y0 = self._y0
            values[0] = A*df.exp(-self._alpha*((x[0] - x0)**2 + (x[1] - y0)**2))

    frequency = 20*1e-3
    amplitude = -800
    x0 = (-46.4676, 63.478)
    alpha = 0.0025
    applied_current = Source(frequency, amplitude, x0, alpha, time_constant, degree=40)

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
        # neumann_boundary_condition=neumann_bc_dict,
        surface_to_volume_factor=Chi,
        membrane_capacitance=Cm
    )
    return brain


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
    for tag in set(brain.cell_tags) - {*zero_tags} - {None}:
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


def get_solver(brain) -> BidomainSplittingSolver:
    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        # valid_cell_tags=(2, 4),
        valid_cell_tags=(2,),
        reload_extension_modules=False
    )

    pde_parameters = CoupledBidomainParameters(
        linear_solver_type="direct"
    )
    # pde_parameters = CoupledBidomainParameters()

    solver = BidomainSplittingSolver(
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )

    vs_prev, *_ = solver.solution_fields()
    vs_prev.assign(brain.cell_model.initial_conditions())
    compute_initial_condirions(brain, vs_prev, {3})

    ######################################################################################
    # I should try to be fancy with the assigning subdomain, rather than write the assign
    # function myself.
    ######################################################################################
    # data_directory = Path.home() / "Documents/Elektrosjokk/data"
    # pial_border = load_array("pial_points.txt", directory=data_directory)
    # wm_border = load_array("white_points.txt", directory=data_directory)
    # ode_solution = np.load(str(Path.home() / "Documents/Elektrosjokk/data/ic_sol.npy"))
    # ode_time = np.load(str(Path.home() / "Documents/Elektrosjokk/data/ic_time.npy"))

    # print("Interpolate ic is very slow!")
    # interpolate_ic(ode_time, ode_solution, vs_prev, [pial_border[:, :2], wm_border[:, :2]], wavespeed=0.03)
    # vfoo, *_ = vs_prev.split(deepcopy=True)
    # with df.XDMFFile(df.MPI.comm_world, "new_meshes/initial_condition_visualisation.xdmf") as fieldfile:
    #     fieldfile.write(vfoo, 0.0)
    # with df.XDMFFile(df.MPI.comm_world, "new_meshes/initial_condition.xdmf") as fieldfile:
    #     fieldfile.write_checkpoint(vfoo, "initial_condition")
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

    field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=10)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("hdf5"), stride_timestep=40*1000)
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

    point_field_spec = FieldSpec(stride_timestep=1)
    for i, centre in enumerate(point_list):
        points = circle_points(radii=[0, 0.1, 0.2, 0.3], num_points=[1, 6, 18, 24], r0=centre)
        saver.add_field(PointField("psd_v{}".format(i), point_field_spec, points))
        saver.add_field(PointField("psd_u{}".format(i), point_field_spec, points))

    for i, centre in enumerate(edge_points):
        points = circle_points(radii=[0, 0.1], num_points=[1, 6], r0=centre)
        saver.add_field(PointField("psd_csf{}".format(i), point_field_spec, points))

    return saver


if __name__ == "__main__":
    brain = get_brain()
    solver = get_solver(brain)

    identifier = simulation_directory(
        parameters={"time": datetime.datetime.now()},
        directory_name=".simulations/test"
    )

    saver = get_saver(brain, identifier)
    v_field_names = ["v", "psd_v0", "psd_v1", "psd_v2", "psd_v3", "psd_v4", "psd_v5"]
    u_field_names = ["u"] + [f"psd_u{i}" for i in range(6)] + [f"psd_csf{i}" for i in range(8)]
    field_names = v_field_names + u_field_names

    tick = time.perf_counter()
    for i, solution_struct in enumerate(solver.solve(0, 1e1, 0.025)):
        print(f"{i} -- {brain.time(0):.5f} -- {solution_struct.vur.vector().norm('l2'):.6e}")

        if saver.update_this_timestep(field_names=["v"], timestep=i, time=brain.time(0)):
            print("saving")
            v, u, *_ = solution_struct.vur.split(deepcopy=True)
            update_dict = {name: v for name in v_field_names}
            update_dict.update({name: u for name in u_field_names})
            saver.update(brain.time, i, update_dict)
    saver.close()
    tock = time.perf_counter()
    print(tock - tick)
