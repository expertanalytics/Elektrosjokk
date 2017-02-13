import pymesh
import scipy

def make_pymesh(points, simplices=None):
    """ Return an instance of Pymesh.Mesh.Mesh

    Parameters
    ----------
    hull_points : array_like
        The coordinates of the mesh surface
    simplices : array_like, optional
        Indices of points forming 2D simplices on the surface of the mesh. If 
        `simplices` is None, triangulate the points using scipy.spatial.ConvexHull 
        (default in None).
        
    Returns
    -------
    pymesh.Mesh.Mesh
    """
    if simplices is None:
        # use ConvexHull surface triangulation
        surf_tri = scipy.spatial.ConvexHull(points) 
        points = surf_tri.points
        simplices = surf_tri.simplices

    return pymesh.form_mesh(points, simplices)


def save_mesh(mesh, name):
    """ Save mesh as stl

    Parameters
    ----------
    mesh : pymesh.Mesh.Mesh
    name : str
        Save mesh as 'name.stl'
    """
    pymesh.save_mesh("{}.stl".format(name), mesh)
