from fenics import *
import numpy as np
from BrainConvexHull import BrainConvexHull
from utils import make_pymesh, save_mesh
from meshing import remesh
import subprocess
import os


def make_surface_meshes(path, input_name, output_name):
    """ Create brain mesh with translated convex hull with subdomains

    Parameters
    ----------
    path : str
        Path to input brain mesh
    input_name : str
        Base name for input files
    outpu_name : str
        Base name for output files
    """

    brain_mesh = Mesh("{path}/{name}.xml".format(**{"path" : path, "name" : input_name}))
    brain_bmesh = BoundaryMesh(brain_mesh, "exterior")
    brain_surface_points = brain_bmesh.coordinates()        # Get coordinates in the surface
    brain_surface_simplices = brain_bmesh.cells()           # Get xml mesh connectivity
    brain_mesh = make_pymesh(brain_surface_points, brain_surface_simplices)
    save_mesh(brain_mesh, output_name)                      # save brain surface as stl
    assert os.path.exists("{}.stl".format(output_name)), "Failed to save inner brain stl"

    bch = BrainConvexHull(brain_surface_points)
    convex_hull_points = bch.compute_hull()                 # Compute convex hull of brain surface
    scaled_hull_points = bch.scale_hull(convex_hull_points, 1.1)    # Translate the hull outwards
    convex_hull_mesh = make_pymesh(scaled_hull_points)        # Use scipy to triangulate surface
    convex_hull_mesh = remesh(convex_hull_mesh, 1.0)        # remesh

    save_mesh(convex_hull_mesh, "convex_{}".format(output_name))     # save as stl
    assert os.path.exists("convex_{}.stl".format(output_name))


def make_volume_mesh(brain_name):
    """ Make a volume mesh from two stl files

    Parameters
    ----------
    brain_name : str
        base name for input and output files
    """
    inner_mesh_name = "Merge '{}.stl';".format(brain_name)          # Merge 'name.stl';
    outer_mesh_name = "Merge 'convex_{}.stl';".format(brain_name)   # Merge 'name.stl';
    header = ["\n".join([inner_mesh_name, outer_mesh_name]) + "\n"]

    with open("brain_template.geo", "r") as f:      # Read geo template
        template = f.readlines()

    tmp_geo_name = "tmp"
    with open(tmp_geo_name + ".geo", "w") as ofile: # Write temporary .geo file
        ofile.write("".join(header + template))

    cmd = "gmsh -3 {}.geo".format(tmp_geo_name)
    print cmd
    result = subprocess.check_output(cmd, shell=True)   # run gmsh
    print result

    cmd = "dolfin-convert {} {}".format(tmp_geo_name + ".msh", brain_name + ".xml")
    result = subprocess.check_output(cmd, shell=True)   # convert to xml

    exists = os.path.exists("{}.xml".format(brain_name)) and \
             os.path.exists("{}_physical_region.xml".format(brain_name)) and \
             os.path.exists("{}_facet_region.xml".format(brain_name))
    assert exists, "Failed to create all xmls"


def make_hdf5(brain_name):
    """ Read the three xmls with mesh, physical volume and physical surfaces, and save as hdf5

    Parameters
    ----------
    brain_name : st
        base name for input and output files
    """

    # Read xmls
    mesh = Mesh("{}.xml".format(brain_name))
    domains = MeshFunction("size_t", mesh, "{}_physical_region.xml".format(brain_name))
    boundaries = MeshFunction("size_t", mesh, "{}_facet_region.xml".format(brain_name))

    # Create hdf5 file
    hdf5_file = HDF5File(mesh.mpi_comm(), "{}.h5".format(brain_name), "w")
    hdf5_file.write(mesh, "/mesh")
    hdf5_file.write(boundaries, "/boundaries")
    hdf5_file.write(domains, "/domains")
    hdf5_file.close()


def clean(brain_name, tmp_name="tmp"):
    """ Remove all intermediary xml, geo and msh files generated in the meshing process

    Parameters
    ----------
    brain_name : str
        base name for xml files
    tmp_name : str, optional
        base name for msh and geo files
    """

    # TODO: Use python remove mmodule to clean up 
    cmd = "rm {0}.xml {0}_physical_region.xml {0}_facet_region.xml".format(brain_name)
    subprocess.check_output(cmd, shell=True)

    cmd = "rm {0}.geo {0}.msh".format(tmp_name)
    subprocess.check_output(cmd, shell=True)


def test_subdomains(path, brain_name, hull_name):
    """ But I use the hdf5 files for everything!
    """
    brain_mesh = Mesh("{0}/{1}.xml".format(path, brain_name))
    hull_mesh = Mesh("{}.xml".format(hull_name))
    facets = MeshFunction("size_t", hull_mesh, "{}_facet_region.xml".format(hull_name))
    cells = MeshFunction("size_t", hull_mesh, "{}_physical_region.xml".format(hull_name))

    brain_ds = Measure("ds", domain=brain_mesh)
    brain_dx = Measure("dx", domain=brain_mesh)
    original_volume = assemble(Constant(1)*brain_dx())
    original_surface = assemble(Constant(1)*brain_ds())

    hull_ds = Measure("ds", domain=hull_mesh, subdomain_data=facets)
    hull_brain_dS = Measure("dS", domain=hull_mesh, subdomain_data=facets)
    hull_dx = Measure("dx", domain=hull_mesh, subdomain_data=cells)
    hull_brain_volume = assemble(Constant(1)*hull_dx(1))
    hull_water_volume = assemble(Constant(1)*hull_dx(2))
    hull_brainsurface0 = assemble(Constant(1)*hull_brain_dS(0))
    hull_brainsurface1 = assemble(Constant(1)*hull_brain_dS(1))
    hull_brainsurface2 = assemble(Constant(1)*hull_ds(2))

    print set(cells.array())
    print set(facets.array())

    print "Original brain"
    print "volume: ", original_volume
    print "surface: ", original_surface
    print
    print "Hull brain"
    print "brain volume: ", hull_brain_volume
    print "water volume: ", hull_water_volume
    print "brain surface0: ", hull_brainsurface0
    print "brain surface1: ", hull_brainsurface1
    print "brain surface2: ", hull_brainsurface2

    #print hull_brain_volume - original_volume
    #assert hull_water_volume > 0
    #assert hull_brain_volume > 0
    #print hull_brainsurface - priginal_surface


def test_poisson(brain_name):
    mesh = Mesh()
    hdf5_file = HDF5File(mesh.mpi_comm(), "{}.h5".format(brain_name), "r")
    hdf5_file.read(mesh, "/mesh", False)

    # Brain volume tag = 7, SAS volume tag = 8
    domains = MeshFunction("size_t", mesh)
    hdf5_file.read(domains, "/domains")

    # Brain surface tag = 1, exterior surface tag = 2
    boundaries = FacetFunction("size_t", mesh)
    hdf5_file.read(boundaries, "/boundaries")

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Boundary condition on exterior boundary
    assert set(domains.array()) == set([1, 2])
    assert set(boundaries.array()) == set([0, 1, 2])

    bcs = DirichletBC(V, Constant(0), boundaries, 2)

    dx = Measure("dx", domain=mesh, subdomain_data=domains)

    print assemble(Constant(1)*dx(7))
    print assemble(Constant(1)*dx(8))

    F = inner(grad(u), grad(v))*dx(1) + inner(grad(u), grad(v))*dx(2) \
        - Constant(1)*v*dx(1) - Constant(2)*v*dx(2)

    a = lhs(F)
    L = rhs(F)

    u_ = Function(V)
    solve(a == L, u_, bcs)

    print np.linalg.norm(u_.vector().array())
    File("test_{}.pvd".format(brain_name)) << u_


if __name__ == "__main__":
    path = "../kent-and-meshes"
    input_name = "erika_res32"
    output_name = "brain"
    make_surface_meshes(path, input_name, output_name)
    print "made surfaces"

    make_volume_mesh(output_name)
    print "made volume"

    test_subdomains(path, input_name, output_name)
    make_hdf5(output_name)
    print "made hdf5"

    print "running poisson"
    test_poisson(output_name)
    #clean(output_name)
