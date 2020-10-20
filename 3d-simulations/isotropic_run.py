import warnings
import datetime
import time
import math
import socket
import argparse

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
    get_indicator_function,
    get_current_time_mpi,
    store_arguments,
    check_bounds,
)

from postspec import (
    FieldSpec,
    SaverSpec,
)

from postfields import (
    Field,
    PointField,
    BoundaryField,
)

import logging


LOG_FILE_NAME = "isotropic_log"
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()

fileHandler = logging.FileHandler(f"{LOG_FILE_NAME}.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


df.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
df.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
df.parameters["form_compiler"]["quadrature_degree"] = 3


def grid(*, N: int, centre: tp.Sequence[float], dx: float) -> np.ndarray:
    dim = len(centre)
    foo = np.zeros([N]*(dim - 1) + [dim])

    for i, pi in enumerate(centre):
        foo[..., i] = np.linspace(pi - dx/2, pi + dx/2, N)

    return foo.reshape(-1, dim)


def filter_grid_points(
    *,
    indicator_function: df.Function,
    centre: tp.Sequence[float],
    N: int,
    dx: float
) -> tp.List[np.ndarray]:
    _indicator = indicator_function
    grid_points = [centre]

    try:
        _target_indicator = int(_indicator(centre))
    except RuntimeError as e:
        logger.info("point centre outside of domain")
        return grid_points

    for _p in grid(N=N, centre=centre, dx=dx):
        try:
            if _indicator(_p) == _target_indicator:
                grid_points.append(_p)
        except RuntimeError as e:
            logger.info("point outisde of domain")
    return grid_points


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


def get_brain(*, mesh_name: str, conductivity: float) -> Model:
    time_constant = df.Constant(0)

    # Realistic mesh
    _hostname = socket.gethostname()
    logger.debug("Hostname: {_hostname}")
    if "debian" in _hostname:
        mesh_directory = Path.home() / "Documents/brain3d/meshes"
    elif "saga" in _hostname:
        mesh_directory = Path("/cluster/projects/nn9279k/jakobes/meshes")
    elif "abacus" in _hostname:
        mesh_directory = Path("/mn/kadingir/opsects_000000/meshes")
    else:
        mesh_directory = Path("meshes")
    logger.info(f"Using mesh directory {mesh_directory}")

    mesh, cell_function = get_mesh(mesh_directory, mesh_name)
    mesh.coordinates()[:] /= 10
    indicator_function = get_indicator_function(mesh_directory / f"{mesh_name}_indicator.xdmf", mesh)

    # 1 -- GM, 2 -- WM
    Mi_dict = {1: conductivity, 2: conductivity, 3: 0}      # 1 mS/cm for WM and GM
    Me_dict = {1: 2.76, 2: 1.26, 3: 17.9}

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
    odemap.add_ode(1, odesolver_module.Cressman(Ks))        # 1 --- Gray matter
    # odemap.add_ode(2, odesolver_module.Cressman(Ku))      # 2 --- White matter
    odemap.add_ode(11, odesolver_module.Cressman(Ku))       # 11 --- Unstable gray matter
    # odemap.add_ode(21, odesolver_module.Cressman(Ku))     # 21 --- Unstable white matter

    splitting_parameters = MultiCellSplittingSolver.default_parameters()
    # cg, petsc_amg runs on saga
    splitting_parameters["BidomainSolver"]["linear_solver_type"] = "iterative"
    splitting_parameters["BidomainSolver"]["algorithm"] = "cg"
    splitting_parameters["BidomainSolver"]["preconditioner"] = "petsc_amg"

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
    *,
    brain: Model,
    outpath: Path,
    point_path_list: tp.Optional[tp.Sequence[Path]]
) -> tp.Tuple[Saver, tp.Sequence[str]]:
    sourcefiles = ["isotropic_run.py"]

    jobscript_path = Path("jobscript_isotropic.slurm")
    if jobscript_path.is_file():
        sourcefiles += [str(jobscript_path)]        # not usre str() is necessary
    store_sourcefiles(map(Path, sourcefiles), outpath)

    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh, brain.cell_domains)

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint",), stride_timestep=100, num_steps_in_part=None)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    if point_path_list is None:
        point_path_list = []

    v_point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=0)
    u_point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=1)
    point_name_list = []

    for point_path in point_path_list:
        points = np.loadtxt(str(point_path))
        points /= 10    # convert to cm
        check_bounds(points, limit=100)        # check for cm
        for point_index, centre in enumerate(points):
            point_name = f"{point_path.stem.split('.')[0].split('_')[-1]}"
            grid_points = filter_grid_points(
                indicator_function=brain.indicator_function,
                centre=centre,
                N=5,
                dx=4
            )

            # V points
            _vp_name = f"{point_name}_points_v{point_index}"
            saver.add_field(PointField(_vp_name, v_point_field_spec, grid_points))
            point_name_list.append(_vp_name)

            # U points
            _up_name = f"{point_name}_points_u{point_index}"
            saver.add_field(PointField(_up_name, u_point_field_spec, grid_points))
            point_name_list.append(_up_name)

    return saver, point_name_list


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-T",
        "--final-time",
        help="Solve the model in the time interval (0, T). Given in ms",
        type=float,
        required=True
    )

    parser.add_argument(
        "-dt",
        "--timestep",
        help="The timestep dt, Given in ms.",
        type=float,
        required=False,
        default=0.025
    )

    parser.add_argument(
        "--mesh-name",
        help="The name of the mesh (excluding suffix). It serves as a prefix to conductivities.",
        type=str,
        required=True
    )

    parser.add_argument(
        "--point-path",
        help="Path to the points used for PointField sampling. Has to support np.loadtxt.",
        nargs="+",
        type=Path,
        required=False,
        default=None
    )

    return parser


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    def run(args) -> None:
        Ks = 4
        Ku = 8
        conductivity = 1

        mesh_name = args.mesh_name
        point_path = args.point_path
        T = args.final_time
        dt = args.timestep

        brain = get_brain(mesh_name=mesh_name, conductivity=conductivity)
        solver = get_solver(brain=brain, Ks=Ks, Ku=Ku)

        current_time = get_current_time_mpi()
        if df.MPI.rank(df.MPI.comm_world) == 0:
            current_time = datetime.datetime.now()
        else:
            current_time = None
        current_time = df.MPI.comm_world.bcast(current_time, root=0)

        _home = Path(".")
        _hostname = socket.gethostname()
        if "abacus" in _hostname:
            _home = Path("/mn/kadingir/opsects_000000/data/simulations")
        identifier = simulation_directory(
            home=_home,
            parameters={
                 "time": current_time,
                 "conductivity": conductivity,
                 "Ks": Ks,
                 "Ku": Ku
            },
            directory_name="brain3d_isotropic"
        )
        store_arguments(args=args, out_path=identifier)

        saver, point_name_list = get_saver(brain=brain, outpath=identifier, point_path_list=point_path)

        tick = time.perf_counter()

        for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve(0, T, dt)):
            norm = vur.vector().norm('l2')
            if math.isnan(norm):
                raise ValueError("Solution diverged")
            logger.info(f"{i} -- {brain.time(0):.5f} -- {norm:.6e}")

            update_dict = dict()
            if saver.update_this_timestep(field_names=["u", "v"], timestep=i, time=brain.time(0)):
                v, u, *_ = vur.split(deepcopy=True)
                update_dict.update({"v": v, "u": u})

            if saver.update_this_timestep(
                    field_names=point_name_list,
                    timestep=i,
                    time=brain.time(0)
            ):
                update_dict.update({_name: vur for _name in point_name_list})

            if len(update_dict) != 0:
                saver.update(brain.time, i, update_dict)


        saver.close()
        tock = time.perf_counter()
        logger.info("Execution time: {:.2f} s".format(tock - tick))

    parser = create_argument_parser()
    args = parser.parse_args()
    run(args)
