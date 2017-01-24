import numpy as np


class ConvHull:
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

        A, B = self._maxdist(S)
        C = self._maxdist_AB(A, B, S)
        D = self._maxdist_ABC(A, B, C, S)
        conv_S += [A, B, C, D]

        n_abc = np.cross(A, np.cross(B, C))

        # first three are a facet, check if the forth point is in negative direction
        facets = [[A, C, B, D],
                  [A, B, D, C],
                  [B, C, D, A],
                  [A, D, C, B]]

        for i in range(4):
            f = self._order_facets(*facets[i])  # TODO: Test should be easy to write. Do it
            Sk = S[np.dot(S - f[0], np.cross(f[1] - f[0], f[2] - f[1])) > 0]
            msg = "Facet sorting definately not working. Sizes are %d and %d" % (S.size, Sk.size)
            assert Sk.size < S.size, msg # Which means there is a problem picking C or D
            conv_S += self.quickhull(Sk)

        return conv_S

    def _maxdist(self, points):
        """ Find the two points with the greatest distance between them
        Warning: Thsi method is extremely memory demanding
        """

        length = points.shape[0]
        points_matrix = np.tile(points, (length, 1)).reshape(length, length, 3) # points are 3D 
        points_matrix -= points_matrix.transpose((1, 0, 2))                 # Swap first two axes
        ind = np.argmax(np.sqrt(np.power(points_matrix, 2, points_matrix).sum(2)))
        return points[ind/length], points[ind%length]

    def _maxdist_AB(self, a, b, c):
        """ Compute the distance from point c to the line between a and b """
        norm = np.linalg.norm
        dist = norm(np.cross(b - a, c - a), axis=1)/norm(b - a)
        C = c[np.argmax(np.abs(dist))]

        assert (C != a).any()
        assert (C != b).any()

        return c[np.argmax(np.abs(dist))]

    def _maxdist_ABC(self, a, b, c, S):
        """ In the set S, find the point most distant to the plane defined by a, b, c """
        # Plane is defied by a and n
        n = np.cross(b - a, c - a)

        dist = np.abs(np.dot(S - a, n))
        p = S[np.argmax(dist)]
        assert (p != a).any(), "({}, {}, {}), ({}, {}, {})".format(p[0], p[1], p[2], a[0], a[1], a[2])
        assert (p != b).any(), "({}, {}, {}), ({}, {}, {})".format(p[0], p[1], p[2], b[0], b[1], b[2])
        assert (p != c).any(), "({}, {}, {}), ({}, {}, {})".format(p[0], p[1], p[2], c[0], c[1], c[2])
        return p

    def _order_facets(self, a, b, c, d):
        """ Order the vertices a, b, c such that ab x ac points away from ad
        """

        direction = np.dot(d - a, np.cross(b - a, c - a))
        assert direction != 0

        if direction < 0:
            return [a, b, c, d]
        else:
            return [a, c, b, d]



if __name__ == "__main__":
    import sys
    print sys.getrecursionlimit()

    N = 1e3
    S = np.random.random((int(N), 3))
    cv = ConvHull()
    conv_S = cv.quickhull(S)
    assert len(conv_S) < conv_S, "Passed reursion, but most likely too many points in conv(S)"
