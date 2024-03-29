import warnings
import datetime
import time
import math
import socket
import argparse

import typing as tp

from pathlib import Path
from collections import namedtuple

import numpy as np
import dolfin as df

from xalbrain import (
    Model,
    MultiCellSplittingSolver,
    MultiCellSolver,
    SplittingSolver,
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
    store_arguments,
    check_bounds,
    solve_IC
)

from postspec import (
    FieldSpec,
    SaverSpec,
)

from postfields import (
    Field,
    PointField,
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


def get_brain(mesh_name: str, mesh_dir: tp.Optional[Path]):
    time_constant = df.Constant(0)

    # Realistic mesh
    if mesh_dir is not None:
        mesh_directory = mesh_dir
    else:
        _hostname = socket.gethostname()
        logger.debug("Hostname: {_hostname}")
        if "debian" in _hostname:
            mesh_directory = Path.home() / "Documents/brain3d/2d-meshes/2d-meshes"
        elif "saga" in _hostname:
            mesh_directory = Path("/cluster/projects/nn9279k/jakobes/2d-meshes")
        elif "abacus" in _hostname:
            mesh_directory = Path("/mn/kadingir/opsects_000000/2d-meshes")
        else:
            mesh_directory = Path("2d-meshes")
    logger.info(f"Using mesh directory {mesh_directory}")

    mesh, cell_function, _ = get_mesh(mesh_directory, mesh_name)
    mesh.coordinates()[:] /= 10
    indicator_function = get_indicator_function(mesh_directory / f"{mesh_name}_indicator.xdmf", mesh)

    # 1 -- GM, 2 -- WM
    # Dougherty et. al. 2014 -- They are not explicit about the anisotropy
    # I will follow the methof from Lee et. al. 2013, but use numbers from Dougherty
    M_i_gray = 1.0      # Dougherty
    # M_i_white = 1.0      # Dougherty
    M_i_white = 5.5     # Dougherty
    logger.info(f"Mi white: {M_i_white}")

    # Doughert et. al 2014
    M_e_gray = 2.78     # Dougherty
    M_e_white = 1.26    # Dougherty
    M_e_white = 5.5    # Dougherty
    logger.info(f"Me white: {M_e_white}")

    CSF = 17.6
    SKULL = 0.1
    SKIN = 4.3
    GRAY_i = M_i_gray

    Mi_dict = {
        1: M_i_white,
        2: M_i_white,
        3: M_i_gray,
        4: M_i_gray,
        5: M_i_gray,
    }
    Me_dict = {
        1: M_e_white,
        2: M_e_white,
        3: M_e_gray,
        4: M_e_gray,
        5: M_e_gray,
    }

    if not "wo" in mesh_name:
        Mi_dict[6] = 1e-4
        Mi_dict[7] = 1e-4
        Mi_dict[8] = 1e-4

        Me_dict[6] = CSF
        Me_dict[7] = SKULL
        Me_dict[8] = SKIN

    brain = Model(
        domain=mesh,
        time=time_constant,
        M_i=Mi_dict,
        M_e=Me_dict,
        cell_models=Cressman(),      # Default parameters
        cell_domains=cell_function,
        indicator_function=indicator_function,
    )
    return brain


def get_solver(*, brain: Model, Ks: float, Ku: float, synaptic: bool) -> MultiCellSplittingSolver:
    odesolver_module = load_module("LatticeODESolver")
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(3, odesolver_module.Cressman(Ks))        # 3 --- Gray matter -- stable
    odemap.add_ode(4, odesolver_module.Cressman(Ku))        # 4 --- Gray matter -- unstable
    odemap.add_ode(5, odesolver_module.Cressman(Ks))        # 5 --- Gray matter -- stable

    if synaptic:
        odemap.add_ode(1, odesolver_module.Synaptic())     # Left white matter
        # odemap.add_ode(2, odesolver_module.Synaptic())     # Right white matter

    # splitting_parameters = SplittingSolver.default_parameters()
    splitting_parameters = MultiCellSplittingSolver.default_parameters()
    splitting_parameters["BidomainSolver"]["linear_solver_type"] = "iterative"
    # splitting_parameters["BidomainSolver"]["linear_solver_type"] = "direct"

    # # gmres
    # ## petsc_amg, ilu,

    # # bicgstab, tfqmr
    # ## ilu
    splitting_parameters["BidomainSolver"]["algorithm"] = "gmres"
    splitting_parameters["BidomainSolver"]["preconditioner"] = "petsc_amg"

    # Physical parameters
    splitting_parameters["BidomainSolver"]["Chi"] = 1.26e3
    splitting_parameters["BidomainSolver"]["Cm"] = 1.0

    # solver = SplittingSolver(model=brain, parameters=splitting_parameters)
    solver = MultiCellSplittingSolver(
        model=brain,
        parameter_map=odemap,
        parameters=splitting_parameters,
    )

    vs_prev, *_ = solver.solution_fields()

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

    if synaptic:
        WHITE_IC = [0]*7                    # voltage + 2 state variables
        WHITE_IC[0] = -6.06953303e+01       # Voltage
        WHITE_IC[1] = 0                     # Hmm
        WHITE_IC[2] = 0                     # Hmmmm
    else:
        WHITE_IC = STABLE_IC

    cell_model_dict = {
        1: WHITE_IC,
        2: WHITE_IC,
        3: STABLE_IC,
        4: UNSTABLE_IC,
        5: STABLE_IC,
    }

    if 6 in brain.cell_domains.array():
        cell_model_dict[6] = WHITE_IC
        cell_model_dict[7] = WHITE_IC
        cell_model_dict[8] = WHITE_IC

    vs_prev.assign(
        solve_IC(brain.mesh, brain.cell_domains, cell_model_dict, dimension=len(UNSTABLE_IC))
    )
    return solver

    # # initial conditions for `vs`
    # CSF_IC = tuple([0]*7)

    # STABLE_IC = (    # stable
    #     -6.70340802e+01,
    #     1.18435132e-02,
    #     7.03013587e-02,
    #     9.78136054e-01,
    #     1.49366709e-07,
    #     3.95901396e+00,
    #     1.78009722e+01
    # )

    # UNSTABLE_IC = (
    #     -6.06953303e+01,
    #     2.63773216e-02,
    #     1.09906468e-01,
    #     9.49154804e-01,
    #     7.69181883e-02,
    #     1.08414264e+01,
    #     1.89251358e+01
    # )

    # WHITE_IC = STABLE_IC

    # cell_model_dict = { #     1: WHITE_IC,
    #     2: WHITE_IC,
    #     3: STABLE_IC,
    #     4: UNSTABLE_IC,
    #     5: STABLE_IC
    # }

    # if 6 in brain.cell_domains.array():
    #     cell_model_dict[6] = CSF_IC

    # odesolver_module.assign_vector(
    #     vs_prev.vector(),
    #     cell_model_dict,
    #     brain.cell_domains,
    #     vs_prev.function_space()._cpp_object
    # )
    # return solver

    # vs_prev.assign(brain.cell_models.initial_conditions())
    # return solver


def get_saver(
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

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint",), stride_timestep=20, num_steps_in_part=None)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint",), stride_timestep=20*1000, num_steps_in_part=None)
    saver.add_field(Field("vs", field_spec_checkpoint))

    v_point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=0)
    u_point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=1)
    point_name_list = []
    for point_path in point_path_list:
        # point_dir = point_path.parent
        # Hmm
        point_name = point_path.stem.split(".")[0].split("_")[-1]

        points = np.loadtxt(str(point_path))
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
        "--mesh-dir",
        type=Path,
        required=False,
        default=None
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

    parser.add_argument(
        "--alpha",
        help="Steepness factor in the conductivity near the GM CSF interface.",
        type=float,
        required=False,
        default=1
    )

    parser.add_argument(
        "--synaptic",
        action="store_true",
        help="Use a passive synaptic model in the white matter."
    )

    return parser


def validate_arguments(args: tp.Any) -> None:
    return True


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    def run(args) -> None:
        Ks = 4
        Ku = 8
        mesh_name = args.mesh_name
        T = args.final_time
        dt = args.timestep

        logger.info(f"mesh name: {args.mesh_name}")
        logger.info(f"Ks: {Ks}")
        logger.info(f"Ku: {Ku}")

        brain = get_brain(mesh_name, mesh_dir=args.mesh_dir)
        solver = get_solver(brain=brain, Ks=Ks, Ku=Ku, synaptic=args.synaptic)

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
                "Ks": Ks,
                "Ku": Ku,
                "mesh-name": mesh_name,
                "dt": dt,
                "T": T,
            },
            directory_name="brain2d_isotropic"
        )
        store_arguments(args=args, out_path=identifier)

        saver, point_name_list = get_saver(brain=brain, outpath=identifier, point_path_list=args.point_path)

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
    # validate_arguments(args)
    run(args)
