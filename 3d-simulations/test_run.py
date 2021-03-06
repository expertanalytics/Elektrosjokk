import warnings
import datetime
import resource
import time
import math

import typing as tp

from pathlib import Path

import numpy as np
import dolfin as df

from xalbrain import (
    Model,
    MultiCellSplittingSolver,
    MultiCellSolver,
    BidomainSolver,
)

from extension_modules import load_module

from xalbrain.cellmodels import Cressman

from post import Saver

from postutils import (
    store_sourcefiles,
    simulation_directory,
    get_mesh,
    get_indicator_function
)

from postspec import (
    FieldSpec,
    SaverSpec,
)

from postfields import (
    Field,
    PointField,
)

df.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
df.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
df.parameters["form_compiler"]["quadrature_degree"] = 3


def get_conductivities(mesh, directory: Path) -> tp.Tuple[df.Function, df.Function]:
    function_space = df.TensorFunctionSpace(mesh, "CG", 1)
    extracellular_function = df.Function(function_space)
    intracellular_function = df.Function(function_space)

    name = "indicator"  #TODO: change to conductivity
    with df.XDMFFile(str(directory / "extracellular_conductivity.xdmf")) as ifh:
        ifh.read_checkpoint(extracellular_function, name, counter=0)

    with df.XDMFFile(str(directory / "intracellular_conductivity.xdmf")) as ifh:
        ifh.read_checkpoint(intracellular_function, name, counter=0)

    return intracellular_function, extracellular_function


def get_brain(*, conductivity: float):
    time_constant = df.Constant(0)

    # Realistic mesh
    # mesh_directory = Path("meshes")
    mesh_directory = Path.home() / "Documents/brain3d/meshes"
    mesh_name = "brain_32"
    mesh, cell_function = get_mesh(mesh_directory, mesh_name)
    mesh.coordinates()[:] /= 10
    indicator_function = get_indicator_function(mesh_directory / f"{mesh_name}_indicator.xdmf", mesh)

    # 1 -- GM, 2 -- WM
    Mi_dict = {1: conductivity, 2: conductivity}      # 1 mS/cm for WM and GM
    Me_dict = {1: 2.76, 2: 1.26}

    brain = Model(
        domain=mesh,
        time=time_constant,
        M_i=Mi_dict,
        M_e=Me_dict,
        cell_models=Cressman(),      # Default parameters
        cell_domains=cell_function,
        indicator_function=indicator_function
    )
    return brain


def get_solver(*, brain: Model, Ks: float, Ku: float) -> MultiCellSplittingSolver:

    odesolver_module = load_module("LatticeODESolver")
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(1, odesolver_module.Cressman(Ks))
    # odemap.add_ode(2, odesolver_module.Cressman(Ku))
    odemap.add_ode(11, odesolver_module.Cressman(Ks))
    # odemap.add_ode(21, odesolver_module.Cressman(Ku))

    splitting_parameters = MultiCellSplittingSolver.default_parameters()
    splitting_parameters["BidomainSolver"]["linear_solver_type"] = "iterative"
    # splitting_parameters["BidomainSolver"]["algorithm"] = "default"
    # splitting_parameters["BidomainSolver"]["preconditioner"] = "petsc_amg"
    splitting_parameters["BidomainSolver"]["algorithm"] = "gmres"
    splitting_parameters["BidomainSolver"]["preconditioner"] = "amg"

    # Physical parameters
    splitting_parameters["BidomainSolver"]["Chi"] = 1.26e3
    splitting_parameters["BidomainSolver"]["Cm"] = 1.0

    solver = MultiCellSplittingSolver(
        model=brain,
        parameter_map=odemap,
        parameters=splitting_parameters,
    )

    vs_prev, *_ = solver.solution_fields()
    vs_prev.assign(brain.cell_models.initial_conditions())
    return solver


def get_saver(
    brain: Model,
    outpath: Path,
) -> Saver:
    sourcefiles = [
        "isotropic_run.py",
    ]
    store_sourcefiles(map(Path, sourcefiles), outpath)

    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh, brain.cell_domains)

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint"), stride_timestep=1)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint"), stride_timestep=1)
    saver.add_field(Field("vs", field_spec_checkpoint))

    points = np.zeros((10, 3))
    for i in range(3):
        points[:, i] = np.linspace(0, 1, 10)
    point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=0)
    saver.add_field(PointField("v_points", point_field_spec, points))

    point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=1)
    saver.add_field(PointField("u_points", point_field_spec, points))
    return saver


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    def run(conductivity, Ks, Ku):
        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        dt = 0.05
        T = 10*dt
        brain = get_brain(conductivity=conductivity)
        solver = get_solver(brain=brain, Ks=Ks, Ku=Ku)

        if df.MPI.rank(df.MPI.comm_world) == 0:
            current_time = datetime.datetime.now()
        else:
            current_time = None
        current_time = df.MPI.comm_world.bcast(current_time, root=0)

        identifier = simulation_directory(
            home=Path("."),
            parameters={
                 "time": current_time,
                 "conductivity": conductivity,
                 "Ks": Ks,
                 "Ku": Ku
            },
            directory_name="brain3d"
        )

        saver = get_saver(brain=brain, outpath=identifier)

        tick = time.perf_counter()

        for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve(0, T, dt)):
            norm = vur.vector().norm('l2')
            if math.isnan(norm):
                assert False, "nan nan nan"
            print(f"{i} -- {brain.time(0):.5f} -- {norm:.6e}")

            update_dict = dict()
            if saver.update_this_timestep(field_names=["u", "v"], timestep=i, time=brain.time(0)):
                v, u, *_ = vur.split(deepcopy=True)
                update_dict.update({"v": v, "u": u})

            if saver.update_this_timestep(field_names=["vs"], timestep=i, time=brain.time(0)):
                update_dict.update({"vs": vs})

            if saver.update_this_timestep(field_names=["v_points", "u_points"], timestep=i, time=brain.time(0)):
                update_dict.update({"v_points": vs, "u_points": vs})

            if len(update_dict) != 0:
                saver.update(brain.time, i, update_dict)

        saver.close()
        tock = time.perf_counter()
        max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
        print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
        print("Execution time: {:.2f} s".format(tock - tick))

    run(1, 4, 8)
