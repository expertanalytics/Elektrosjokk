import pymesh
import numpy as np
from numpy.linalg import norm



def fix_mesh(mesh):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    target_len = 1.0

    print "Target resolution: {} mm".format(target_len)
    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices

    count = 0
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len, preserve_feature=False)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)     # TODO: Determine max angle 

        if mesh.num_vertices == num_vertices:
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

    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 100)   # TODO: Determine max angle
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh




if __name__ == "__main__":
    mesh = pymesh.load_mesh("brain_hull.stl")
    mesh = fix_mesh(mesh)
    pymesh.save_mesh("test_remesh_notpreserve.stl", mesh)

    pymesh.timethis.summarize()

