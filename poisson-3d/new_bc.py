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

import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main():
    casedir = Path(".")
    casename = "e9591a1d"
    hull_mesh_name = "brain_128.xml"
    loader_spec = LoaderSpec(casedir=casedir / casename)

    loader = Loader(loader_spec)
    hull_mesh = df.Mesh(hull_mesh_name)
    logger.info(f"Read hull mesh: {hull_mesh_name}")

    function_space_hull = df.FunctionSpace(hull_mesh, "CG", 1)
    u = df.TrialFunction(function_space_hull)
    v = df.TestFunction(function_space_hull)

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.Constant(1)*v*df.dx

    A = df.assemble(a)
    b = df.assemble(L)

    solver_type = "cg"
    preconditioner_type = "petsc_amg"
    logger.info(f"Setting up Poisson solver with ({solver_type}, {preconditioner_type})")
    solver = df.KrylovSolver(solver_type, preconditioner_type)
    solver.set_operator(A)

    solution_function = df.Function(function_space_hull)

    facet_function = df.MeshFunction("size_t", hull_mesh, hull_mesh.geometry().dim() - 1)

    mesh_function_directory = Path.home() / "Documents/brain3d/skull_mesh"
    mesh_function_name = "test_mf.xdmf"
    with df.XDMFFile(str(mesh_function_directory / mesh_function_name)) as mf_file:
        mf_file.read(facet_function)

    current_time = get_current_time_mpi()

    identifier = simulation_directory(
        home=Path("."),
        parameters={
            "time": current_time,
            "casename": casename,
            "hull_mesh_name": hull_mesh_name
        },
        directory_name="brain3d"
    )

    # TODO: look at outpath name
    outpath = identifier

    source_files = ["new_bc.py"]
    store_sourcefiles(map(Path, source_files), outpath)
    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(hull_mesh, facet_function)

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint"), stride_timestep=1)
    saver.add_field(Field("u_poisson", field_spec_checkpoint))

    for timestep, (time, brain_ue) in enumerate(loader.load_checkpoint("u")):        # u is the extracellular potential
        logger.debug(f"time, {time, brain_ue.vector().norm('l2')}")

        # interpolate between meshens
        bc_func = interpolate_nonmatching_mesh(brain_ue, function_space_hull)
        logger.debug(f"Boundary condition norm: {time, bc_func.vector.norm('l2')}")

        # FIXME: Insert mesh function here!
        boundary_condition = df.DirichletBC(function_space_hull, bc_func, facet_function, 1)
        # boundary_condition = df.DirichletBC(function_space_hull, bc_func, df.DomainBoundary())
        boundary_condition.apply(A, b)

        solver.solve(solution_function.vector(), b)
        logger.info(time, solution_function.vector().norm("l"))

        # TODO: add a post.Saver and so on
        # TODO: Store the time in some manner

        saver.update(time, timestep, {"u_poisson": solution_function})


    saver.close()
