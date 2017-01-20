import numpy as np


class ConvSet:
    def __init__(self):
        """
        self.S = S
        self.dim = S.shape[1]
        assert self.dim == 3, "Only support  3D"
        """

    def quickhull(self, S):
        if S.shape[0] <= 4:
            return list(S)

        conv_S = []
        # Extremals in xdir        
        xA = S[np.argmax(S[:, 0])]
        xB = S[np.argmin(S[:, 0])]

        yA = S[np.argmax(S[:, 1])]
        yB = S[np.argmin(S[:, 1])]

        zA = S[np.argmax(S[:, 2])]
        zB = S[np.argmin(S[:, 2])]
        extremals = np.array([xA, xB, yA, yB, zA, zB])

        # TODO: Write tests for the maxdist functions. at least one is probalby wrong
        A, B = self._maxdist(extremals)
        C = self._maxdist_AB(A, B, extremals)
        D = self._maxdist_ABC(A, B, C, S)
        conv_S += [A, B, C, D]

        n_abc = np.cross(A, np.cross(B, C))

        facets = [[A, C, B, D],
                  [A, B, D, C],
                  [A, D, C, B],
                  [B, D, C, A]]

        for i in range(4):
            f = self._order_facets(*facets[i])  # TODO: Test should be easy to write. Do it
            # TODO: Check that A, B, C, D are removed from Sk

            Sk = S[np.dot(S, np.cross(f[0], np.cross(f[1], f[2]))) > 0]
            msg = "Facet sorting definately not working. Sizes are %d and %d" % (S.size, Sk.size)
            assert Sk.size < S.size, msg # Which means there is a problem picking C or D
            conv_S += self.quickhull(Sk)

        return conv_S

    def _maxdist(self, points):
        """ Find the two points with the greatest distance between them

        """

        length = len(points)
        points_matrix = np.tile(points, (length, 1)).reshape(length, length, 3) # points are 3D 
        points_matrix -= points_matrix.transpose((1, 0, 2))     # Swap first two axes
        ind = np.argmax(np.abs(points_matrix).sum(2))   # L2 norm over each distance

        m = np.abs(points_matrix).sum(2)    # for testing
        assert abs((points[ind/length] - points[ind%length])).sum() == m.max()

        return points[ind/length], points[ind%length]

    def _maxdist_AB_test(self, a, b, c):
        """ Compute the distance from point c to the line between a and b """

        n_abc = np.cross(b - a, c - a)      # Normal to abc-plane
        n_ab = np.cross(b - a, n_abc - a)   # Normal to ab in abc-plane
        return np.dot(c - a, n_ab)          # Orthogonal coponent of c

    def _maxdist_ABC(self, a, b, c, S):
        """ In the set S, find the point most distant to the plane defined by a, b, c """
        n = np.cross(a, np.cross(b, c))
        # Plane is defied by a and n

        dist = np.abs(np.dot(S - a, n))
        point = S[np.argmax(dist)]
        for p in (a, b, c):
        assert (point =! a).any()
        assert (point =! b).any()
        assert (point =! c).any()
        return point

    def _order_facets(self, a, b, c, d):
        """ return ordered list of a, b, c, d such that
        np.dot(d, np.cross(a, np.cross(b, c)))
        """
        direction = np.dot(d, np.cross(a, np.cross(b, c)))
        if direction > 0:
            return [a, b, c, d]
        else:
            return [a, c, b, d]

    def _maxdist_AB(self, a, b, c):
        """ Compute the distance from point c to the line between a and b """
        assert isinstance(c, np.ndarray)

        # FIXME: This is the most probable source of error

        n_abc = np.cross(b - a, c - a)                  # Normal to abc-plane
        n_ab = np.cross(b - a, n_abc)               # Normal to ab in abc-plane
        dist = np.einsum("ij, ij->i", c, n_ab)      # TODO: What does this really do?
        C = c[np.argmax(np.abs(dist))]
        assert (C != a).any()
        assert (C != b).any()
        return c[np.argmax(np.abs(dist))]


if __name__ == "__main__":
    import sys
    print sys.getrecursionlimit()
    # TODO: Write tests for each method to find the error
    # TODO: Write visualization, look at 3d method

    #! Possibly an infinite recursion?
    N = 1e1
    S = np.random.random((int(N), 3))
    cv = ConvSet()
    conv_S = cv.quickhull(S)
    assert conv_S.size < conv_S, "Passed reursion, but most likely too many points in conv(S)"
    print conv_S
    print "conv_S", conv_S

