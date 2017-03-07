import pytest
import numpy as np
from BrainConvexHull import BrainConvexHull


def test_scale_hull(bch):
    """Assert that each point in the hull is moved away from the centre. I dont't know by how much,
    as it is each facet which is moved.
    """
    delta = 1.1     # Distance to move each point

    hull = bch.compute_hull()
    centre = np.mean(hull, axis=0)
    scaled_hull = bch.scale_hull(hull, delta)

    hull_dist = np.linalg.norm(hull - centre, axis=1)  # The distance from the hull to the centre
    scaled_hull_dist = np.linalg.norm(scaled_hull - centre, axis=1)

    # Assert that the distance to the centre has increased
    assert (scaled_hull_dist - hull_dist > 0).all()


def test_move_point(bch):
    p1 = np.array([2., 3., 4.])

    n = np.array([2., 2., 2.])
    nf = np.sqrt(12)
    n /= nf
    dist = 4
    number_of_simplices = 3

    expected = np.array([2 + 8./3/nf, 3 + 8./3/nf, 4 + 8./3/nf])

    bch._move_point(p1, n, dist, number_of_simplices)
    assert np.linalg.norm(p1 - expected) < 1e-15


def test_compute_distance_to_brain(bch, S):
    delta = 1.1     # Distance to move each point

    computed = bch.compute_distance_to_brain(S*delta)
    assert (np.abs(computed - (delta - 1)) < 1e-15).all()


@pytest.fixture(scope="module")
def S():
    """ Use unit sphere as test space
    """
    N = 10
    r = 1
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    points = np.dstack((x, y, z))
    return points[0]


@pytest.fixture(scope="module")
def bch(S):
    return BrainConvexHull(S)


if __name__ == "__main__":
    test_compute_distance_to_brain(S())
    test_move_point(bch(S()))
    test_scale_hull(bch(S()))
