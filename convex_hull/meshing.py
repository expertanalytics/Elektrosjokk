import pymesh


def remesh(mesh, target_length, preserve_features=False, abs_treshold=1e-6, maxiter=10):
    """Remesh input mesh.

    Parameters
    ----------
    mesh : pymesh.Mesh.Mesh
    target_length : float
        Split all edges longer than `target_length`
    preserve_features : bool, optional
        True if shape features should be preserved (default is False)
    abs_treshold : float, optional
        Collapse all edges with length equal to or below `abs_threshold` (default is 1e-6).
    maxiter : int, optional
        Maximum number of iterations, (default is 10).

    Returns
    -------
    pymesh.Mesh.Mesh

    Notes
    -----
    This script is slightly modified form
    (PyMesh documentation)[https://github.com/qnzhou/PyMesh/blob/master/scripts/fix_mesh.py]
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
            # TODO: Add another stopping criterion
            break

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
