import typing as tp

import argparse

from pathlib import Path

import dolfin as df
import numpy as np


def read_mesh(mesh_path: Path) -> tp.Tuple[df.Mesh, df.MeshFunction]:
    directory = mesh_path.parent

    _mesh_name = f"{mesh_path.stem}.xdmf"
    _cf_name = f"{mesh_path.stem}_cf.xdmf"

    # read mesh
    mesh = df.Mesh()
    with df.XDMFFile(str(directory / _mesh_name)) as mesh_file:
        mesh_file.read(mesh)

    # read cell function
    cell_function = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
    with df.XDMFFile(str(directory / _cf_name)) as cf_file:
        cf_file.read(cell_function)

    cell_tags = np.unique(cell_function.array()).astype(np.int_)
    print(f"Cell tags: {cell_tags}")
    return mesh, cell_function


def indicator_function(
    mesh: df.Mesh,
    cell_function: df.MeshFunction,
    cell_tags: tp.Iterable
) -> df.Function:
    dX = df.Measure("dx", domain=mesh, subdomain_data=cell_function)

    DG_fs = df.FunctionSpace(mesh, "DG", 1)

    u = df.TrialFunction(DG_fs)
    v = df.TestFunction(DG_fs)

    sol = df.Function(DG_fs)
    sol.vector().zero()     # Make sure it is initialised to zero

    # NB! For some reason map(int, cell_tags) does not work with the cell function.
    F = 0
    F += -u*v*dX(1) + df.Constant(1)*v*dX(1)
    F += -u*v*dX(2) + df.Constant(2)*v*dX(2)

    a = df.lhs(F)
    L = df.rhs(F)

    # TODO: Why keep diagonal and ident_zeros?
    A = df.assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = df.assemble(L)
    solver = df.KrylovSolver("cg", "petsc_amg")
    solver.set_operator(A)
    solver.solve(sol.vector(), b)

    sol.vector()[:] = np.rint(sol.vector().get_local())
    print(np.unique(sol.vector().get_local()))
    return sol


def assign_indicator_function(mesh, cell_function):
    from IPython import embed
    function_space = df.FunctionSpace(mesh, "CG", 1)
    dofmap = function_space.dofmap()

    indicator_function = df.Function(function_space)

    cell_tags = np.sort(np.unique(cell_function.array()))

    for tag in cell_tags:
        sub_cell_indices = np.where(cell_function.array() == tag)[0]

        for cell_index in sub_cell_indices:
            dofs = dofmap.cell_dofs(cell_index)
            foo = cell_function.array()[cell_index]
            indicator_function.vector()[dofs] = foo

    return indicator_function


def save_function(indicator_function: df.Function, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with df.XDMFFile(str(output_path)) as xdmf:
        xdmf.write_checkpoint(indicator_function, "indicator", 0)


def read_function(mesh: df.Mesh, name: Path) -> df.Function:
    function_space = df.FunctionSpace(mesh, "DG", 1)
    function = df.Function(function_space)

    with df.XDMFFile(str(name)) as xdmf:
        xdmf.read_checkpoint(function, "indicator", 0)
    return function


if __name__ == "__main__":
    mesh_name = Path("mesh/brain_64.xdmf")
    mesh, cell_function = read_mesh(mesh_name)

    indicator = assign_indicator_function(mesh, cell_function)
    # indicator = indicator_function(mesh, cell_function, (1, 2))

    mesh_directory = mesh_name.parent
    indicator_name = f"{mesh_name.stem}_indicator.xdmf"
    save_function(indicator, mesh_directory / indicator_name)

    function = read_function(mesh, mesh_directory / indicator_name)
    df.File("indicator.pvd") << function
