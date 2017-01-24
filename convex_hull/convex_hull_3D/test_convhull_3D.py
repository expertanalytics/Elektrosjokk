from convhull_3D import ConvHull
import pytest
import numpy as np
import scipy.spatial


def test_maxdist(S, cv):
    A = [-2, -1, 0]
    B = [2, -1, 0]
    cA, cB = cv._maxdist(S)

    successA = (cA == A).all() or (cA == B).all()
    successB = (cB == B).all() or (cB == A).all()
    assert (cA != cB).any()
    assert successA and successB

    pass

def test_maxdist_AB(S, cv):
    A = np.array([-2, -1, 0])
    B = np.array([2, -1, 0])
    cC = cv._maxdist_AB(A, B, S)

    C = np.array([0, 2, 0])
    assert (cC == C).all()

def test_maxdist_ABC(S, cv):
    A = np.array([-2, -1, 0])
    B = np.array([2, -1, 0])
    C = np.array([0, 2, 0])
    D = np.array([0, 0, 1])

    cD = cv._maxdist_ABC(A, B, C, S)
    assert (cD == D).all()

def test_order_facets(S, cv):
    # Give order_facets a list which will return negative and expect positive
    # Give order_facets a list which will return positive and expect positive
    # Give lots of random lists

    A = np.array([-2, -1, 0])
    B = np.array([2, -1, 0])
    C = np.array([0, 2, 0])
    D = np.array([0, 0, 1])

    facets = np.array([[A, C, B, D],
                       [A, B, D, C],
                       [B, C, D, A],
                       [A, D, C, B]])

    new_facets = np.array([cv._order_facets(*points) for points in facets])
    assert (new_facets == facets).all()
    """
    for i in xrange(4):
        print facets[i]
        print new_facets[i]
        print
    """

def test_equal_scipy(cv, S):
    """ lazy. Test sim of sets are equal """
    myresult = cv.quickhull(S)
    hull = scipy.spatial.ConvexHull(S)
    scipyresult = S[hull.vertices]
    assert np.sum(myresult) == np.sum(scipyresult)


@pytest.fixture(scope="module")
def S():
    """ Return test data set, but is it a good test set?
    """
    N = 10
    A = [-2, -1, 0]
    B = [2, -1, 0]
    C = [0, 2, 0]
    D = [0, 0, 1]
    extremals = np.array([A, B, C, D])

    phi = np.linspace(0, 2*np.pi, N)
    costheta = np.linspace(-1, 1, N)
    r = 0.5*np.linspace(0, 1, N)

    theta = np.arccos(costheta)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    S = np.zeros((N + 4, 3))
    for i, coor in enumerate((x, y, z)):
        S[:, i] = np.concatenate((coor, extremals[:, i]))

    assert (S[np.argmax(S[:, 0])] == B).all()
    assert (S[np.argmax(S[:, 1])] == C).all()
    assert (S[np.argmax(S[:, 2])] == D).all()
    assert (S[np.argmin(S[:, 0])] == A).all()
    return S


@pytest.fixture(scope="module")
def cv():
    return ConvHull()


if __name__ == "__main__":
    test_order_facets(S(), cv())
