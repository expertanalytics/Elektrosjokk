r"""Module for the computation and scaling of a convex hull using the scipy wrapper
of QHull.

The scaling is accomplished by computing the normal vector for each simplex and moving
each vertex a distance :math: `frac{\delta}{N_s}`, where :math:`N_s` are the number of
simplices in which that vertex appears.
"""

from timeit import default_timer as timer
import numpy as np
import scipy.spatial


__all__ = ["BrainConvexHull"]


class BrainConvexHull:
    """Class for computing and scaling a convec hull.

    Class for computing the

    Parameters:
        points  : array_like of dimension (n, d), d > 2

    Attributes:
        brain_points : The input non-convex set

    Methods:
        compute_hull()
            Compute the convex hull
        scale_hull(hull_coordinates, delta)
            Translate the convex hull a distance delta outwards
        make_mesh(hull_points, meshname)
            Save hull_points as stl
        compute_distance_to_brain(scaled_hull_points)
            Compute the minimum distance to the original set
    """

    def __init__(self, brain_points):
        self.brain_points = brain_points

    def compute_hull(self):
        """Compute the convex hull.

        Returns:
            numpy.ndarray
                The vertices forming the convex hull of `self.brain_points`
        """
        hull = scipy.spatial.ConvexHull(self.brain_points)
        return self.brain_points[hull.vertices]

    def _move_point(self, point, n, dist, number_of_simplices):
        """Move the array `point` a distance `dist`/`number_of_simplices` in direction
        n.

        Parameters:
            point               : numpy.ndarray
            n                   : array_like
                                  Same dimension as `point`
            dist                : int or float
            number_of_simplices : int
        """
        assert abs(np.linalg.norm(n) - 1.0) < 1e-15
        point += n*float(dist)/number_of_simplices

    def scale_hull(self, hull_coordinates, delta, time=True):
        """Move each facet a distance delta in it's putward noraml direction.

        Parameters:
            hull_coordinates    : array_like of dimension (n, d), d > 2
            delta               : int/float
                                  The distance to move each facet in the triangulation
                                  of `hull_coordinates`.

        Returns:
            numpy.ndarray
                A translated copy of `hull_coordinates`
        """
        # Compute convex hull for triangulation
        hull = scipy.spatial.ConvexHull(hull_coordinates)
        face_indices = hull.simplices

        num_faces_pr_point = np.bincount(face_indices.flatten())
        # Each vertex is part of at least 3 simplices
        assert (num_faces_pr_point >= 3).all()

        # Compute geometric centre
        geometric_center = np.mean(hull.points[hull.vertices], axis=0)

        new_points = hull.points.copy()

        tic = timer()   # timing python for loop
        for simplex in face_indices:    # A simplex consists of three point indices
            v0, v1, v2 = hull.points[simplex]
            n = np.cross(v1 - v0, v2 - v0)
            if np.dot(v0 - geometric_center, n) < 0:  # make sure we have the outward normal
                n = np.cross(v2 - v0, v1 - v0)
            n /= np.linalg.norm(n)                         # Normalise n

            for point_index in simplex:
                # Get the number of simplices this point is part of
                number_of_simplices = num_faces_pr_point[point_index]

                # move the point delta/(num_surrounding_simplices)*normal_vector
                self._move_point(new_points[point_index], n, delta, number_of_simplices)

        toc = timer() - tic
        if time:
            print "\033[1;37;31m{}\033[0m".format("Time spent scaling hull: {}s".format(toc))

        return new_points

    def compute_distance_to_brain(self, scaled_hull_points):
        """Compute the minimum distance from the convex hull to the point set.

        This is a wrapper method for scipy.spatial.cKDTree       

        Parameters:
            ----------
            scaled_hull_points : array_like
                                 Array of same inner dimension as `scaled_hull_points`.
        """
        tree = scipy.spatial.cKDTree(self.brain_points)

        # Returns tuple(distance, index)
        return tree.query(scaled_hull_points, k=1)[0]


if __name__ == "__main__":
    points = np.load("surface_points.npy")
    bch = BrainConvexHull(points)
    hull_points = bch.compute_hull()
    scaled_hull = bch.scale_hull(hull_points, 1.1)
    bch.compute_distance_to_brain(scaled_hull)
