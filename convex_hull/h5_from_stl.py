import subprocess
import os
import numpy as np
from dolfin import Mesh, MeshFunction, HDF5File


def make_volume_mesh(brain_name):
    """Make a volume mesh from two stl files

    Parameters
    ----------
    brain_name : str
        base name for input and output files
    """
    inner_mesh_name = "Merge '{}.stl';".format(brain_name)          # Merge 'name.stl';
    outer_mesh_name = "Merge 'convex_{}.stl';".format(brain_name)   # Merge 'name.stl';
    header = ["\n".join([inner_mesh_name, outer_mesh_name]) + "\n"]

    with open("brain_template.geo", "r") as geo_template:      # Read geo template
        template = geo_template.readlines()

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
    """Read the three xmls with mesh, physical volume and physical surfaces, and save as hdf5

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


if __name__ == "__main__":
    output_name = "brain"
    make_volume_mesh(output_name)
    make_hdf5(output_name)
