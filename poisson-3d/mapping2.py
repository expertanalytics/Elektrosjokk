import dolfin as df
import numpy as np

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

from scipy.spatial import cKDTree

from fenicstools import interpolate_nonmatching_mesh

import time
import resource
import argparse
import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def assign_point_array(recarray, ndarray, dtype) -> None:
    for i, (name, _) in enumerate(dtype):
        recarray[name] = ndarray[:, i]


def convert_to_point_array(array: np.ndarray) -> np.ndarray:
    point_dtype = [("p0", "f8"), ("p1", "f8"), ("p2", "f8")]
    new_array = np.zeros(array.shape[0], dtype=point_dtype)
    new_array["p0"] = array[:, 0]
    new_array["p1"] = array[:, 1]
    new_array["p2"] = array[:, 2]
    return new_array


def main(mesh_path, skull_mesh_path):
    mesh = df.Mesh()
    with df.XDMFFile(str(mesh_path)) as infile:
        infile.read(mesh)
    mesh.coordinates()[:] /= 10
    print(mesh.coordinates())

    skull_mesh_directory = skull_mesh_path.parent
    skull_mesh_name = skull_mesh_path.stem

    skull_mesh = df.Mesh()
    with df.XDMFFile(str(skull_mesh_path)) as infile:
        infile.read(skull_mesh)
    skull_mesh.coordinates()[:] /= 10
    print(skull_mesh.coordinates())

    cell_function_path = skull_mesh_directory / f"{skull_mesh_name}_cf.xdmf"
    logger.info(f"Reading facet function: {cell_function_path}")
    cell_mvc = df.MeshValueCollection("size_t", skull_mesh, skull_mesh.geometry().dim())
    with df.XDMFFile(str(cell_function_path)) as infile:
        infile.read(cell_mvc)
    cell_function = df.MeshFunction("size_t", skull_mesh, cell_mvc)

    facet_function_path = skull_mesh_directory / f"{skull_mesh_name}_ff.xdmf"
    logger.info(f"Reading facet function: {facet_function_path}")
    facet_mvc = df.MeshValueCollection("size_t", skull_mesh, skull_mesh.geometry().dim() - 1)
    with df.XDMFFile(str(facet_function_path)) as infile:
        infile.read(facet_mvc)
    facet_function = df.MeshFunction("size_t", skull_mesh, facet_mvc)

    # Creating form and assembling system
    function_space_hull = df.FunctionSpace(skull_mesh, "CG", 1)

    print("---------------------------------------------------------------------------------------")
    # check that all points in bmesh are in skull_mesh
    bmesh = df.BoundaryMesh(mesh, "exterior")
    skull_points = convert_to_point_array(skull_mesh.coordinates())
    bmesh_points = convert_to_point_array(bmesh.coordinates())

    with df.XDMFFile("bmesh.xdmf") as of:
        of.write(bmesh)

    with df.XDMFFile("skull_mesh_ff.xdmf") as of:
        of.write(facet_function)

    bmesh_in_skull_mesh = np.in1d(bmesh_points, skull_points)
    print(bmesh_in_skull_mesh.any())
    print(bmesh_in_skull_mesh.all())

    original_function_space = df.FunctionSpace(mesh, "CG", 1)

    orig_dof_coordinates = original_function_space.tabulate_dof_coordinates()
    dof_points = convert_to_point_array(orig_dof_coordinates)

    # Why am I not using a function space defined on bmesh?
    coordinate_point = convert_to_point_array(bmesh.coordinates())
    boundary_dof_points_indices = np.in1d(dof_points, coordinate_point)

    # TODO: I don't think recarray will help me here
    skull_dof_coordinates = function_space_hull.tabulate_dof_coordinates()

    tree = cKDTree(skull_dof_coordinates)
    index_list = []
    # TODO: use original_function_space.tabulate_dof_coordinates()[boundary_dof_points_indices]
    import time
    for point in orig_dof_coordinates[boundary_dof_points_indices]:
        radius, index = tree.query(point)
        index_list.append(index)
        # index_where_equal = np.where(np.isclose(skull_dof_coordinates, point, atol=1e-5))[0]
        # index_list.append(index_where_equal)
        # print(index_where_equal)

    # orig_func = df.project(df.Expression("x[0]*x[1]*x[2]", degree=1), original_function_space)
    orig_func = df.Function(original_function_space)
    orig_func.vector()[:] = 1

    bv = df.Function(function_space_hull)
    df.LagrangeInterpolator.interpolate(bv, orig_func)

    submesh = df.SubMesh(skull_mesh, cell_function, 6)
    SBV = df.FunctionSpace(submesh, "CG", 1)
    sub_bv = df.Function(SBV)

    df.LagrangeInterpolator.interpolate(sub_bv, orig_func)

    # bv = df.Function(function_space_hull)
    # print("--------------------------------------------------")
    # bmesh_copy = bv.vector().get_local()
    # bmesh_copy[index_list] = orig_func.vector().get_local()[boundary_dof_points_indices]

    # bv.vector()[:] = bmesh_copy

    # print(bv.vector().get_local())
    # print("--------------------------------------------------")


    with df.XDMFFile("bmesh.xdmf") as of:
        of.write(bmesh)

    with df.XDMFFile("submesh.xdmf") as of:
        of.write(submesh)

    with df.XDMFFile("bfunc.xdmf") as of:
        of.write(bv)

    with df.XDMFFile("subfunc.xdmf") as of:
        of.write(sub_bv)

    with df.XDMFFile("ref.xdmf") as of:
        of.write(orig_func)

    print("---------------------------------------------------------------------------------------")


if __name__ == "__main__":
    mesh_dir = Path("/home/jakobes/Documents/brain3d/test")
    skull_path = mesh_dir / "skull_32.xdmf"
    mesh_path = mesh_dir / "brain_32.xdmf"
    main(mesh_path, skull_path)
