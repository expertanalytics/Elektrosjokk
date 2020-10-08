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


LOG_FILE_NAME = "anisotropic_log"
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

CONDUCTIVITY_TUPLE = namedtuple("CONDUCTIVITY_TUPLE", ["intracellular", "extracellular"])


def get_conductivities(
    *,
    mesh: df.Mesh,
    mesh_name: str,
    mesh_directory: Path,
    anisotropy_type: str
) -> tp.Tuple[df.Function, df.Function]:
    function_space = df.TensorFunctionSpace(mesh, "DG", 0)
    extracellular_function = df.Function(function_space)
    intracellular_function = df.Function(function_space)

    extracellular_file_name = f"{mesh_directory / mesh_name}_intracellular_conductivity_{anisotropy_type}.xdmf"
    intracellular_file_name = f"{mesh_directory / mesh_name}_extracellular_conductivity_{anisotropy_type}.xdmf"

    name = "conductivity"  #TODO: change to conductivity
    with df.XDMFFile(str(extracellular_file_name)) as ifh:
        ifh.read_checkpoint(extracellular_function, name, counter=0)

    with df.XDMFFile(str(intracellular_file_name)) as ifh:
        ifh.read_checkpoint(intracellular_function, name, counter=0)

    conductivity_tuple = CONDUCTIVITY_TUPLE(
        intracellular=intracellular_function,
        extracellular=extracellular_function
    )
    return conductivity_tuple


def get_brain(mesh_name: str, anisotropy_type: str):
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
    conductivity_tuple = get_conductivities(
        mesh=mesh,
        mesh_name=mesh_name,
        mesh_directory=mesh_directory,
        anisotropy_type=anisotropy_type
    )

    # Dougherty et. al. 2014 -- They are not explicit about the anisotropy
    # I will follow the methof from Lee et. al. 2013, but use numbers from Dougherty
    M_i_gray = 1.0      # Dougherty

    # Doughert et. al 2014
    M_e_gray = 2.78     # Dougherty
    # M_e_white = 1.26    # Dougherty

    # From "A guideline for head volume conductor modeling in EEG and MEG -- Vorwerk et. al 2014"
    # CSF = 17.6
    # skull = 0.1
    # skin = 4.3

    CSF = 1.0
    skull = 1.0
    skin = 1.0

    MI_dict = {
        # 2: conductivity_tuple.intracellular,
        2: 1,
        1: M_i_gray,
        3: 1e-4,
        4: 1e-4,
        5: 1e-4,
        6: 1e-4
    }

    ME_dict = {
        # 2: conductivity_tuple.extracellular,
        2: 1.26,
        1: M_e_gray,
        3: CSF,
        4: skin,
        5: skull,
        6: CSF
    }

    stimulus_dict = {
        1: df.Constant(1)
    }


    brain = Model(
        domain=mesh,
        time=time_constant,
        M_i=MI_dict,
        M_e=ME_dict,
        cell_models=Cressman(),      # Default parameters
        cell_domains=cell_function,
        indicator_function=indicator_function,
        stimulus=stimulus_dict
    )
    return brain


def get_solver(*, brain: Model, Ks: float, Ku: float) -> MultiCellSplittingSolver:
    odesolver_module = load_module("LatticeODESolver")
    # Indices are in reference to indicator_function, not cell_function
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(1, odesolver_module.Cressman(Ks))        # 1 --- Gray matter
    # odemap.add_ode(2, odesolver_module.Cressman(Ku))      # 2 --- White matter
    odemap.add_ode(11, odesolver_module.Cressman(Ku))       # 11 --- Unstable gray matter
    # odemap.add_ode(21, odesolver_module.Cressman(Ku))     # 21 --- Unstable white matter

    splitting_parameters = MultiCellSplittingSolver.default_parameters()
    splitting_parameters["BidomainSolver"]["linear_solver_type"] = "iterative"

    # # gmres
    # ## petsc_amg, ilu,

    # # bicgstab, tfqmr
    # ## ilu
    splitting_parameters["BidomainSolver"]["algorithm"] = "gmres"
    splitting_parameters["BidomainSolver"]["preconditioner"] = "petsc_amg"

    # Physical parameters
    splitting_parameters["BidomainSolver"]["Chi"] = 1.26e3
    splitting_parameters["BidomainSolver"]["Cm"] = 1.0

    splitting_parameters["apply_stimulus_current_to_pde"] = True

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
    point_path_list: tp.Optional[tp.Sequence[Path]]
) -> tp.Tuple[Saver, tp.Sequence[str]]:
    sourcefiles = ["anisotropic_run.py"]
    jobscript_path = Path("jobscript_anisotropic.slurm")
    if jobscript_path.is_file():
        sourcefiles += [str(jobscript_path)]        # not usre str() is necessary

    store_sourcefiles(map(Path, sourcefiles), outpath)

    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh, brain.cell_domains)

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint",), stride_timestep=20, num_steps_in_part=None)
    saver.add_field(Field("v", field_spec_checkpoint))
    saver.add_field(Field("u", field_spec_checkpoint))

    if point_path_list is None:
        point_path_list = []

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
        "--anisotropy",     # dummy, constant, dti
        help="The type of anisotropy. Fits the ending of a conductivity function." \
        "Supported: 'dti', 'constant', 'dummy'",
        type=str,
        required=True,
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


def validate_arguments(args: tp.Any) -> None:
    valid_anisotropy = {"dti", "constant", "dummy"}
    if not args.anisotropy in valid_anisotropy:
        raise ValueError(f"Unknown anisotropy: expects {valid_anisotropy}")


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    # def run(*, Ks: float, Ku: float, mesh_name: str, dt: float, T: float, anisotropy: str) -> None:
    def run(args) -> None:
        Ks = 4
        Ku = 8
        mesh_name = args.mesh_name
        anisotropy = args.anisotropy
        T = args.final_time
        dt = args.timestep

        logger.info(f"mesh name: {args.mesh_name}")
        logger.info(f"Ks: {Ks}")
        logger.info(f"Ku: {Ku}")
        brain = get_brain(mesh_name, anisotropy)
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
                "Ks": Ks,
                "Ku": Ku,
                "mesh-name": mesh_name,
                "dt": dt,
                "T": T,
                "anisotropy": anisotropy
            },
            directory_name="brain3d_head"
        )
        store_arguments(args=args, out_path=identifier)

        saver, point_name_list = get_saver(
            brain=brain,
            outpath=identifier,
            point_path_list=args.point_path
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

            # if saver.update_this_timestep(field_names=["vs"], timestep=i, time=brain.time(0)):
            #     update_dict.update({"vs": vs})

            # if saver.update_this_timestep(field_names=["v_points", "u_points"], timestep=i, time=brain.time(0)):
            if saver.update_this_timestep(
                    field_names=point_name_list,
                    timestep=i,
                    time=brain.time(0)
            ):
                # update_dict.update({"v_points": vs, "u_points": vs})
                update_dict.update({_name: vs for _name in point_name_list})

            if len(update_dict) != 0:
                saver.update(brain.time, i, update_dict)

        saver.close()
        tock = time.perf_counter()
        logger.info("Execution time: {:.2f} s".format(tock - tick))

    parser = create_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    run(args)
