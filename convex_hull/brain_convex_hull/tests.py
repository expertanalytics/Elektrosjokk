import pytest
import numpy as np
import scipy.spatial
from read_brain import *


def test_compute_and_scale_delaunay(S):
    """ Compute correct result by hand
    """
    pass


def test_compute_and_scale_uniform_volume(S, tol=1e-5):
    """ Volume should increase by factor**3
        Eps was chosen arbitrarily
    """
    volume_before = 4./3*np.pi*1**3

    #factor = 1.1**3
    factor = 1.1
    S = compute_and_scale_uniform(S, factor=factor)

    volume_after = volume_before*factor
    print volume_after
    print volume_before*factor
    assert np.abs(volume_after - volume_before*factor**3) < tol


def test_compute_and_scale_uniform(S):
    """ Compute correct result by hand
            The only difference is that r changes by factor
            i.e. each component increases by factor
    """
    factor = 1.1

    print S.shape
    convS = compute_and_scale_uniform(S, factor=factor)
    print convS.shape





@pytest.fixture(scope="module")
def S():
    """ Use unitsphere as test space
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

if __name__ == "__main__":
    test_compute_and_scale_uniform_volume(S())
