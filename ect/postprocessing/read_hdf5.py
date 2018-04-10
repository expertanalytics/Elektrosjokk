import dolfin

from typing import (
    Tuple
)


#info(f.parameters, True)


def read_hdf5_mesh(filename: str, field="/Mesh") -> dolfin.Mesh:
    """Read the mesh from a hdf5 file."""
    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), filename, "r") as hdf5file:
        hdf5file.read(mesh, field, False)
    return mesh


def read_hdf5_functions(
        mesh: dolfin.Mesh,
        filename: str,
        fields: Tuple[str],
        family="CG",
        degree=1
) -> dolfin.Function:
    """Read functions from a hdf5file."""
    V = dolfin.FunctionSpace(mesh, family, degree)
    v = dolfin.Function(V)
    with dolfin.HDF5File(mesh.mpi_comm(), filename, "r") as hdf5file:
        for name in fields:
            hdf5file.read(v, name)
            yield v
