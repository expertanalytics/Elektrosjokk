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
    solve_IC,
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


def get_brain(
    *,
    mesh_name: str,
    mesh_dir: tp.Optional[Path],
    unstable_tags: tp.Sequence[int]
) -> Model:
    time_constant = df.Constant(0)

    # Realistic mesh

    if mesh_dir is not None:
        mesh_directory = mesh_dir
    else:
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

    mesh, cell_function, _ = get_mesh(mesh_directory, mesh_name)
    mesh.coordinates()[:] /= 10
    indicator_function = get_indicator_function(mesh_directory / f"{mesh_name}_indicator.xdmf", mesh)

    M_i_gray = 1.0      # Dougherty
    M_e_gray = 2.78     # Dougherty

    MI_dict = {
        2: 1,
        1: M_i_gray,
        3: 1e-4,
    }
    ME_dict = {
        2: 1.26,
        1: M_e_gray,
        3: 17.6,
    }

    for tag in unstable_tags:
        ME_dict[tag] = M_e_gray
        MI_dict[tag] = M_i_gray

    brain = Model(
        domain=mesh,
        time=time_constant,
        M_i=MI_dict,
        M_e=ME_dict,
        cell_models=Cressman(),      # Default parameters
        cell_domains=cell_function,
        indicator_function=indicator_function
    )

    return brain


def get_solver(
    *,
    brain: Model,
    Ks: float,
    Ku: float,
    ic_type: str,
    unstable_tags: tp.Sequence[int],
    cressman_white: bool
) -> MultiCellSplittingSolver:
    odesolver_module = load_module("LatticeODESolver")
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(1, odesolver_module.Cressman(Ks))        # 1 --- Gray matter
    for key in unstable_tags:
        odemap.add_ode(key, odesolver_module.Cressman(Ku))

    if cressman_white:
        odemap.add_ode(2, odesolver_module.Cressman(Ks))        # 2 --- White matter

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
    if ic_type == "cressman":
        vs_prev.assign(brain.cell_models.initial_conditions())
        return solver
    elif ic_type == "stable_unstable":
        # initial conditions for `vs`
        CSF_IC = tuple([0]*7)

        STABLE_IC = (    # stable
            -6.70340802e+01,
            1.18435132e-02,
            7.03013587e-02,
            9.78136054e-01,
            1.49366709e-07,
            3.95901396e+00,
            1.78009722e+01
        )

        UNSTABLE_IC = (
            -6.06953303e+01,
            2.63773216e-02,
            1.09906468e-01,
            9.49154804e-01,
            7.69181883e-02,
            1.08414264e+01,
            1.89251358e+01
        )

        WHITE_IC = STABLE_IC        # This alkso works for `cressman_white`

        cell_model_dict = {
            1: STABLE_IC,
            2: WHITE_IC,
            3: WHITE_IC,
        }

        for key in unstable_tags:
            cell_model_dict[key] = UNSTABLE_IC

        vs_prev.assign(
            solve_IC(brain.mesh, brain.cell_domains, cell_model_dict, dimension=len(UNSTABLE_IC))
        )
        return solver
    else:
        raise RuntimeError("Unknon ic type")


def get_saver(
    *,
    brain: Model,
    outpath: Path,
    point_dir: tp.Optional[Path]
) -> tp.Tuple[Saver, tp.Sequence[str]]:
    sourcefiles = ["isotropic_run.py"]
    jobscript_path = Path("jobscript_isotropic.slurm")
    if jobscript_path.is_file():
        sourcefiles += [str(jobscript_path)]        # not usre str() is necessary
    store_sourcefiles(map(Path, sourcefiles), outpath)

    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh, brain.cell_domains)

    field_spec_checkpoint = FieldSpec(
        save_as=("checkpoint",),
        stride_timestep=400,
        num_steps_in_part=None
    )
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    v_point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=0)
    u_point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=1)
    point_name_list = []

    _hostname = socket.gethostname()
    logger.debug("Hostname: {_hostname}")
    if point_dir is not None:
        _point_directory = point_dir
    elif "debian" in _hostname:
        _point_directory = Path.home() / "Documents/brain3d/points/chosen_points"
    elif "saga" in _hostname:
        _point_directory = Path("/cluster/projects/nn9279k/jakobes/points/chosen_points")
    elif "abacus" in _hostname:
        _point_directory = Path("/mn/kadingir/opsects_000000/points/chosen_points")
    else:
        _point_directory = Path("points/chosen_points")
    logger.info(f"Using point directory {_point_directory}")

    for point_file in _point_directory.iterdir():
        point_name = point_file.stem
        points = np.loadtxt(str(point_file))
        points /= 10    # convert to cm
        check_bounds(points, limit=100)        # check for cm

        # V points
        _vp_name = f"{point_name}_points_v"
        saver.add_field(PointField(_vp_name, v_point_field_spec, points))
        point_name_list.append(_vp_name)

        # U points
        _up_name = f"{point_name}_points_u"
        saver.add_field(PointField(_up_name, u_point_field_spec, points))
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
        type=Path,
        required=False,
        default=None
    )

    parser.add_argument(
        "--mesh-dir",
        type=Path,
        required=False,
        default=None
    )

    parser.add_argument(
        "--cressman",
        action="store_true",
        help="Use standrd cressman initial conditions everywhere."
    )

    parser.add_argument(
        "--stable-unstable",
        action="store_true",
        help="Use stble and unstable initial conditions."
    )

    parser.add_argument(
        "--unstable-tags",
        nargs="+",
        type=int,
        help="Cell tags of the unstable domain.",
        required=True
    )
    return parser


def validate_arguments(args: tp.Any) -> None:
    if args.cressman and args.stable_unstable:
        raise RuntimeError("'cressman' and 'stable-unstable' cannot be set at the same time")
    elif not args.cressman and not args.stable_unstable:
        raise RuntimeError("Either 'cressman' or 'stable-unstable' must be set")


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

        brain = get_brain(
            mesh_name=mesh_name,
            mesh_dir=args.mesh_dir,
            unstable_tags=args.unstable_tags
        )

        if args.cressman:
            ic_type = "cressman"
        elif args.stable_unstable:
            ic_type = "stable_unstable"
        solver = get_solver(
            brain=brain,
            Ks=Ks,
            Ku=Ku,
            ic_type=ic_type,
            unstable_tags=args.unstable_tags,
            cressman_white=args.cressman_white
        )

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

        saver, point_name_list = get_saver(
            brain=brain,
            outpath=identifier,
            point_dir=point_path
        )

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
        logger.info(f"Identifier: {identifier}")

    parser = create_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    run(args)
