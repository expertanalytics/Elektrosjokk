from dolfin import Mesh, BoundaryMesh


def save_bmesh_coordinates(fname="test_surface.xml"):
    """ Read dolfin-mesh and save coordinates as numpy arra

    Parameters
    ----------
    fname : String
            The name of the mesh to read, including file extension
    """
    mesh = Mesh(fname)
    bmesh = BoundaryMesh(mesh, "exterior")
    np.save("test", bmesh.coordinates())
