"""Base class for RK4."""

from typing import Tuple

import numpy as np

from numba import float64


# Usage: jitclass(JITSPEC)(ODESolver)
JITSPEC = [
    ("t_array", float64[:]),
    ("y_array", float64[:, :])
]


class RK4Solver:
    """RK4 solver for (vector) ODEs."""

    def __init__(self, ic: Tuple[float64], T: float, dt: float) ->  None:
        """
        Store intiial condition, end time and time step.

        NB! ic must have a length. Try/Except braks numba nopython mode.

        Args:
            ic: Initial condition of shape 12. The parameters come in the
                following order: V, m, n,, h, NKo, NKi, NNao, NNai, NClo, NCli, Voli, O.
            T: End time.
            dt: Time step.

        The solver will create an array in [0, T] og size int(T/dt).
        """
        self.t_array = np.linspace(0, T, int(T/dt))
        self.y_array = np.zeros(shape=(self.t_array.size, len(ic)))
        self.y_array[0] = ic

    def _rhs(self, t: float, y: float):
        raise NotImplementedError

    def _step(self, y: float, t0: float, t1: float) -> np.ndarray:
        dt = t1 - t0
        k1 = self._rhs(t0, y)
        k2 = self._rhs(t0 + dt/2, y + k1*dt/2)
        k3 = self._rhs(t0 + dt/2, y + k2*dt/2)
        k4 = self._rhs(t0 + dt, y + dt*k3)
        return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    def solve(self) -> None:
        """Solve the ODE."""
        for i in range(1, self.t_array.size):
            t0 = self.t_array[i - 1]
            t1 = self.t_array[i]
            y = self.y_array[i - 1]
            self.y_array[i] = self._step(y, t0, t1)

    @property
    def solution(self) -> np.ndarray:
        """
        Return the solution array.

        NB! Will be zeros apart form the initial condition unless `solve` has been called.
        """
        return self.y_array

    @property
    def time(self) -> np.ndarray:
        """Return the time steps."""
        return self.t_array


if __name__ == "__main__":
    def exact(t: np.ndarray) -> np.ndarray:
        """Return exact solution."""
        return np.exp(-t)*5

    class Test(RK4Solver):
        """Test class."""

        def _rhs(self, t: float, y: float) -> np.ndarray:
            return -y

    solver = Test((5.0,), 1.0, 1e-1)
    solver.solve()
    computed = solver.solution
    t_array = solver.time
    exact_sol = exact(t_array)

    print(np.max(np.abs(computed.flatten() - exact_sol)))
