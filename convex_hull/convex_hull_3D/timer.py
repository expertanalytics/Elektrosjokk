from convhull_3D import ConvHull
import scipy.spatial
from timeit import default_timer as timer
import numpy as np


def my_qhull(S):
    cv = ConvHull()
    return cv.quickhull(S)


def scipy_qhull(S):
    hull = scipy.spatial.ConvexHull(S)
    return S[hull.vertices]


def conv_hull_timer(method, S):
    start = timer()
    result = method(S)
    stop = timer()

    print "{}\t\033[1;37;31m{}\033[0m".format(method.__name__, stop - start)
    return result


if __name__ == "__main__":
    import sys
    #N = int(sys.argv[1])
    N = int(1e4)
    S = np.random.random((int(N), 3))
    for method in (my_qhull, scipy_qhull):
        conv_hull_timer(method, S)
