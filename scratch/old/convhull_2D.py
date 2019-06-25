import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.path import Path
import matplotlib.patches as patches


class ConvSet:  # {{{
    def __init__(self, S):  # {{{
        self.S = S
        self.dim = S.shape[1]
        assert self.dim == 2, "Only support  2D"
        # }}}

    def quickhull(self):    # {{{
        S = self.S
        if self.dim == 2:
            assert S.shape[0] > 2, "S needs at least 2 points"

        A = S[np.argmax(S[:, 0])]
        B = S[np.argmin(S[:, 0])]

        S1 = S[self._outside(A, B, S) > 0]
        S2 = S[self._outside(A, B, S) <= 0]

        # Is there a more elegant way than using lists?
        # I could only operate on S and its indices
        conv_S = [A]
        conv_S += self._findHull(S1, A, B)
        conv_S += [B]
        conv_S += self._findHull(S2, B, A)
        conv_S += [A]
        return np.array(conv_S)      # }}}

    def _findHull(self, Sk, P, Q):     # {{{
        """ I need to draw this to make sure the line method works """
        conv_S = []
        if len(Sk) == 0:
            return conv_S

        C = Sk[np.argmax(self._outside(P, Q, Sk))]

        S1 = Sk[self._outside(P, C, Sk) > 0]
        S2 = Sk[self._outside(C, Q, Sk) > 0]

        conv_S += self._findHull(S1, P, C)
        conv_S += [C]
        conv_S += self._findHull(S2, C, Q)
        return conv_S   # }}}

    def _outside(self, a, b, c):    # {{{
        """ Compute the distance from point c to the line between a and b """
        rot = np.array([[0, -1],
                        [1,  0]])   # Rotation matrix for pi/2 in positive direction
        n = np.dot(rot, b - a)      # rotate ab pi/2 to get a normal vector
        return np.dot(c - a, n)     # return component of ac orthogonal to ab
        # }}}

    def plot(self, verts):   # {{{
        assert isinstance(verts, np.ndarray)
        assert verts.ndim == 2, "Only support 2D for now"

        codes = [Path.LINETO]*verts.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        path = Path(verts, codes)

        fig = mpl.figure()
        ax = fig.add_subplot(111)

        xs, ys = zip(*verts)
        ax.plot(xs, ys, '--', lw=2, color='black', ms=10)
        ax.plot(S[:, 0], S[:, 1], ".")

        # TODO: compute dim from conv_S
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        mpl.show()  # }}}
    # }}}


if __name__ == "__main__":
    N = 1e2
    S = np.random.random((int(N), 2))    # Vector of points in 2D
    cv = ConvSet(S)
    conv = cv.quickhull()
    cv.plot(conv)
