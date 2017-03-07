"""A collection of functions for creating and remeshing stl meshes."""

import pymesh
import scipy

def make_pymesh(points, simplices=None):
    """Create a pymesh mesh instance.

    The point set is assumed to be convex if simplices is not porvided.

    Parameters:
        hull_points (array_like): The coordinates of the mesh surface
        simplices (array_like, optional): Indices of points forming 2D simplices on the 
            surface of the mesh. If `simplices` is None, triangulate the points using 
            scipy.spatial.ConvexHull. Defaults to None.
                    
    Returns:
        pymesh.Mesh.Mesh
    """
    if simplices is None:
        # use ConvexHull surface triangulation
        surf_tri = scipy.spatial.ConvexHull(points)
        points = surf_tri.points
        simplices = surf_tri.simplices

    return pymesh.form_mesh(points, simplices)


def save_mesh(mesh, name):
    """Save pymesh mesh as stl

    Parameters:
        mesh (pymesh.Mesh.Mesh)
        name (str): Save mesh as 'name.stl'
    """
    pymesh.save_mesh("{}.stl".format(name), mesh)


def remesh(mesh, target_length, preserve_features=False, abs_treshold=1e-6, maxiter=10):
    """Remesh input mesh.

    Parameters:
        mesh (pymesh.Mesh.Mesh)
        target_length (float): Split all edges longer than `target_length`
        preserve_features (bool, optional): True if shape features should be preserved
            (default is False)
        abs_treshold (float, optional): Collapse all edges with length equal to or 
            below `abs_threshold`. Defaul to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults is 10.

    Returns:
        pymesh.Mesh.Mesh

    TODO:
        Find a better stopping criterion

    Notes:
        This function is as slightly modified form of
        (https://github.com/qnzhou/PyMesh/blob/master/scripts/fix_mesh.py).
    """
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _ = pymesh.split_long_edges(mesh, target_length)
    num_vertices = mesh.num_vertices    # Used as stopping criterion

    count = 0   # Keep track of number of iterations
    while True:
        mesh, _ = pymesh.collapse_short_edges(mesh, abs_treshold)
        mesh, _ = pymesh.collapse_short_edges(mesh, target_length,
                                              preserve_feature=preserve_features)
        mesh, _ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)     # TODO: Determine max angle 

        if mesh.num_vertices == num_vertices:
            # Is this actually very clever?
            break # TODO: Add another stopping criterion

        num_vertices = mesh.num_vertices
        print "#v: {}".format(num_vertices)
        if count > 20:
            break
        count += 1
        print "iteration: ", count

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)

    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_obtuse_triangles(mesh, 179.0, 100)   # TODO: Determine max angle
    mesh, _ = pymesh.remove_isolated_vertices(mesh)

    return mesh


if __name__ == "__main__":
    mesh = pymesh.load_mesh("brain_hull.stl")

    mesh = remesh(mesh, 1.0)
    pymesh.save_mesh("test_remesh_notpreserve.stl", mesh)

    pymesh.timethis.summarize()
