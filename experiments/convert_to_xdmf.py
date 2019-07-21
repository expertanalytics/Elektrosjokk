""" Write an xdmf file for use in FEniCS.

Thanks to J. S. Dokken for help with this script.

https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/9
"""


import meshio
from collections import namedtuple


MeshData = namedtuple("MeshData", ("points", "cells", "lines", "cell_data", "facet_data"))


def unify_cell_tags(cell_tags, unify):
    """Do nothing."""
    if not unify:
        return
    return


def unify_facet_tags(facet_tags, unify):
    if not unify:
        return
    facet_tags[facet_tags == 9] = 8
    facet_tags[facet_tags == 7] = 6
    facet_tags[facet_tags == 5] = 4


def parse_gmsh_mesh(mesh: meshio.Mesh, unify=False, dim=2) -> MeshData:
    """Extract data structures from meshio. This is gmsh specific."""
    if unify:
        print("Unifying tags according to rule.")

    _points = mesh.points[:, :dim]
    _cells = {"triangle": mesh.cells["triangle"]}
    try:
        _lines = {"line": mesh.cells["line"]}
    except KeyError:
        _lines = None
    try:
        if unify:
            unify_cell_tags(mesh.cell_data["triangle"]["gmsh:physical"], unify)
        _cell_data = {"triangle": {"cell_data": mesh.cell_data["triangle"]["gmsh:physical"]}}
    except KeyError:
        _cell_data = None
    try:
        if unify:
            unify_facet_tags(mesh.cell_data["line"]["gmsh:physical"], unify)
        _facet_data = {"line": {"facet_data": mesh.cell_data["line"]["gmsh:physical"]}}
    except:
        _facet_data = None
    return MeshData(_points, _cells, _lines, _cell_data, _facet_data)


def write_mesh(points, cells, ofname):
    _new_mesh = meshio.Mesh(points=points, cells=cells)
    meshio.write(f"{ofname}.xdmf", _new_mesh)


def write_mesh_function(points, cells, cell_data, ofname):
    _new_mesh = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
    meshio.write(f"{ofname}.xdmf", _new_mesh)


def test_fenics_read(directory, name):
    import dolfin as df

    mesh = df.Mesh()
    with df.XDMFFile(f"{directory}/{name}.xdmf") as infile:
        infile.read(mesh)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(f"{directory}/{name}_mf.xdmf") as infile:
        infile.read(mvc, "cell_data")
    cf = df.MeshFunction("size_t", mesh, mvc)

    mvc = df.MeshValueCollection("size_t", mesh, 1)
    with df.XDMFFile(f"{directory}/{name}_ff.xdmf") as infile:
        infile.read(mvc, "facet_data")
    ff = df.MeshFunction("size_t", mesh, mvc)

    dx_custom = df.Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=2)
    print(df.assemble(1*dx_custom))

    ds_custom = df.Measure("ds", domain=mesh, subdomain_data=ff, subdomain_id=12)
    print(df.assemble(1*ds_custom))


if __name__ == "__main__":
    from pathlib import Path

    mesh_dircetory = Path("new_slice_experiments/idealised_meshes")

    input_name = "new_slice_experiments/new_meshes/skullgmwm_fine.msh"
    output_name = "new_slice_experiments/new_meshes/skullgmwm_fine"

    # for i in range(1, 5):
    # input_name = mesh_dircetory / "idealised{}.msh".format(i)
    # print(input_name)
    # output_name = mesh_dircetory / "idealised{}".format(i)
    mesh = meshio.read(str(input_name))
    mesh_data = parse_gmsh_mesh(mesh, unify=False)

    write_mesh(mesh_data.points, mesh_data.cells, output_name)
    write_mesh_function(mesh_data.points, mesh_data.cells, mesh_data.cell_data, f"{output_name}_cf")
    write_mesh_function(mesh_data.points, mesh_data.lines, mesh_data.facet_data, f"{output_name}_ff")

    # test_fenics_read("foo", "test")
