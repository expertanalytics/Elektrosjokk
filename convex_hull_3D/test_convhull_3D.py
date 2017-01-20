from convhull_3D import ConvSet
import pytest
import numpy as np


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

    C = np.array([0, 1, 0])
    assert (cC == C).all()

def test_maxdist_ABC(S, cv):
    A = np.array([-2, -1, 0])
    B = np.array([2, -1, 0])
    C = np.array([0, 1, 0])
    D = np.array([0, 0, 1])

    cD = cv._maxdist_ABC(A, B, C, S)
    assert (cD == D).all()

def test_order_facets(S, cv):
    pass

@pytest.fixture(scope="module")
def S():
    """ Return test data set, but is it a good test set?
    """
    N = 100
    A = [-2, -1, 0]
    B = [2, -1, 0]
    C = [0, 1, 0]
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
    return ConvSet()


if __name__ == "__main__":
    S()
    test_maxdist(S(), cv())
