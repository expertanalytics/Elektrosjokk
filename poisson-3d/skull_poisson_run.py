import dolfin as df

from pathlib import Path

from post import Loader, Saver

from postspec import (
    LoaderSpec,
    SaverSpec,
    FieldSpec,
)

from postfields import (
    Field,
    PointField,
)

from postutils import (
    store_sourcefiles,
    simulation_directory,
    get_current_time_mpi,
)

from fenicstools import interpolate_nonmatching_mesh

import time
import resource
import argparse
import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--casedir",
        help="Path to the case directory",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--skull",
        help="Path to the skull mesh.",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--brain-tag",
        help="The tag of the inner skull boundary. The brain boundary.",
        required=True,
        type=int
    )

    parser.add_argument(
        "--facet-function",
        help="""Path to the facet function corresponding to the skull mesh. Looks in skull directory
        by default""",
        required=False,
        default=None,
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)

    casedir = args.casedir.parent
    casename = args.casedir.stem

    loader_spec = LoaderSpec(casedir=casedir / casename)
    loader = Loader(loader_spec)

    # Read mesh and mesh function
    logger.info(f"Reading skull mesh: {args.skull}")
    skull_mesh = df.Mesh()
    with df.XDMFFile(str(args.skull)) as hull_mesh_file:
        hull_mesh_file.read(skull_mesh)

    if args.facet_function is None:
        facet_function_path = args.skull.parent / f"{args.skull.stem}_mf.xdmf"
        if not facet_function_path.exists():
            raise FileNotFoundError(
                f"Could not find {facet_function_path}. Please specify manually."
            )
    else:
        facet_function_path = args.facet_function

    logger.info(f"Reading facet function: {facet_function_path}")
    mvc = df.MeshValueCollection("size_t", skull_mesh, skull_mesh.geometry().dim())
    with df.XDMFFile(str(facet_function_path)) as infile:
        infile.read(mvc)
    facet_function = df.MeshFunction("size_t", skull_mesh, mvc)

    # Creating form and assembling system
    function_space_hull = df.FunctionSpace(skull_mesh, "CG", 1)
    u = df.TrialFunction(function_space_hull)
    v = df.TestFunction(function_space_hull)

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.Constant(0)*v*df.dx

    # Boundary condition defined later
    A = df.assemble(a)
    b = df.assemble(L)

    solver_type = "cg"
    preconditioner_type = "petsc_amg"
    logger.info(f"Setting up Poisson solver with ({solver_type}, {preconditioner_type})")
    solver = df.KrylovSolver(solver_type, preconditioner_type)
    solver.set_operator(A)

    solution_function = df.Function(function_space_hull)
    current_time = get_current_time_mpi()

    outpath = simulation_directory(
        home=Path("."),
        parameters={
            "time": current_time,
            "casename": casename,
            "hull_mesh_name": args.skull
        },
        directory_name="skull3d"
    )

    source_files = ["skull_poisson_run.py"]
    store_sourcefiles(map(Path, source_files), outpath)
    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(skull_mesh, facet_function)

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint"), stride_timestep=1)
    saver.add_field(Field("u_poisson", field_spec_checkpoint))

    tick = time.perf_counter()
    for timestep, (solution_time, brain_ue) in enumerate(loader.load_checkpoint("u")):        # u is the extracellular potential
        norm = solution_time, brain_ue.vector().norm("l2")
        logger.debug(f"timestep: {timestep} --- time, {norm}")

        # interpolate between meshens
        bc_func = interpolate_nonmatching_mesh(brain_ue, function_space_hull)
        logger.debug(f"Boundary condition norm: {solution_time, bc_func.vector().norm('l2')}")

        logger.info(f"Applying boundary condition to tag {args.brain_tag}")
        boundary_condition = df.DirichletBC(
            function_space_hull,
            bc_func,
            facet_function,
            args.brain_tag
        )
        boundary_condition.apply(A, b)

        solver.solve(solution_function.vector(), b)
        norm = solution_function.vector().norm("l2")
        logger.info(f"timestep: {timestep} --- time: {solution_time}, norm: {norm}")

        saver.update(solution_time, timestep, {"u_poisson": solution_function})

    tock = time.perf_counter()
    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    logger.info("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    logger.info("Execution time: {:.2f} s".format(tock - tick))
    saver.close()


if __name__ == "__main__":
    main()
