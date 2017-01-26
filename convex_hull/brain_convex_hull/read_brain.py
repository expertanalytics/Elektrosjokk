from dolfin import *
import numpy as np
import scipy.spatial
from stl import mesh


def read(fpath, fname="test_surface.xml"):
    """ Read brain xml mesh and dump the surface-mesh to file """
    mesh = Mesh(fpath)
    bmesh = BoundaryMesh(mesh, "exterior")


def save_coordinates(fname):
    """ Read bmesh and save coordinates as numpy array """
    mesh = Mesh(fname)
    coor_array = mesh.coordinates()
    np.save("test", coor_array)


def scale_point(A, x, b):
    x -= b
    np.dot(A, x)
    x += b
    return x


def compute_and_scale_uniform(points, factor=1):
    """ Compute convex hull of the set 'points',
    """
    hull = scipy.spatial.ConvexHull(points)
    hull_points = hull.points[hull.vertices]
    hull.close()

    # Compute geometric centre
    centroid = np.mean(points[hull.vertices, :], axis=0)

    # Scale the distance from point p to the centre, memory intensive 
    scaling_matrix = np.eye(3)*factor

    for i in xrange(hull_points.shape[0]):
        hull_points[i] = scale_point(scaling_matrix, hull_points[i], centroid)

    return hull_points


def compute_delaunay(points):
    """ Compute the Delaunay triangulation of points, and return vertices
    """
    points = points
    #tri = scipy.spatial.Delaunay(points)

    hull = scipy.spatial.ConvexHull(points)
    vertices = points[hull.simplices]
    return vertices


def save_stl(vertices):
    #cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    cube = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    cube.vectors[:] = vertices
    print "vertices, ", vertices.shape
    print "expected, ", cube.vectors.shape

    cube.save('brain.stl')


def compute_and_scale_delaunay(points, delta=0):
    """ Should I return delaunay triangulation? """
    hull = scipy.spatial.ConvexHull(points)
    tri = scipy.spatial.Delaunay(hull.points[hull.vertices])
    hull.close()
    # (indices, indptr). The indices of neighboring vertices of vertex k are indptr[indices[k]:indices[k+1]]

    indices, indptr = tri.vertex_neighbor_vertices
    points = tri.points

    for k in xrange(points.shape[0]):
        avg_normal = points[indptr[indices[k]:indices[k+1]]].sum(0)
        avg_normal /= np.linalg.norm(avg_normal)
        points[k] += delta*avg_normal
    return points


if __name__ == "__main__":
    #read("../../Elektrosjokk_meshes/erika_res32.xml")
    #save_coordinates("test_surface.xml")
    points = np.load("test.npy")
    hull = compute_and_scale_uniform(points)
    vertices = compute_delaunay(hull)
    save_stl(vertices)
