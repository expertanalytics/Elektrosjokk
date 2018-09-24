"""Utility functions for creating a square puls forcing term."""

import dolfin as df
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from math import cos, sin, pi
from numpy import exp, ndarray

from typing import Tuple, Callable

from pathlib import Path

from hashlib import sha1

import numpy as np


class Iext_rhs:
    """Right hand ide for the periodic forcing term."""

    def __init__(self, period: float) -> None:
        """Store the frequency.

        Args:
            period: The period.
        """
        self.omega = 2*pi / period

    def __call__(self, t: float, y: ndarray) -> Tuple[float, float]:
        """
        return the right hand side of the system of odes.

        Arguments:
            t: The current time.
        """
        u, w = y

        du = u*(1 - u*u - w*w) - self.omega*w
        dw = w*(1 - u*u - w*w) + self.omega*u
        return du, dw


def get_I_ext(interval: float, amplitude: float, period: float, duration: float) -> Callable:
    """Return an interpolation of the forcing term.

    TODO: Rewrite this into a class, and set new IC as last end of period.

    Solve the system of ODEs associated with the forcing frequency, then apply the
    phase change and scale the amplidude.

    Arguments:
        interval: The interval in which to solve the ODEs.
        amplitude: The amplitude.
        period: The period of the forcing.
        duraion: The duration of each pulse in the forcing.

    Returns:
        A callable linear interpolation of the forcing.
    """
    phi = duration*pi/period            # phase change
    interval = (0, interval)            # time interval
    y0 = (1, 0)                         # Initial conditions
    stimulus_rhs = Iext_rhs(period)

    m = sha1()
    m.update(bytes("{}".format(interval), "utf-8"))
    m.update(bytes("{}".format(amplitude), "utf-8"))
    m.update(bytes("{}".format(period), "utf-8"))
    m.update(bytes("{}".format(duration), "utf-8"))
    cache_name = m.hexdigest()

    cache_dir = Path("cache")
    if not cache_dir.exists():
        cache_dir.mkdir()
    if m not in list(cache_dir.iterdir()):
        solver = solve_ivp(stimulus_rhs, interval, y0, vectorized=True, method="BDF")
        u_values, w_values = solver.y
        time_values = solver.t
        np.save(str(cache_dir / cache_name), np.vstack((time_values, u_values, w_values)))
    else:
        time_values, u_values, w_values = np.load("{}.npy".format(cache_dir / cache_dir))

    solver = solve_ivp(stimulus_rhs, interval, y0, vectorized=True, method="BDF")
    u_values, w_values = solver.y

    I_ext = amplitude/(1 + exp(100*((1 - u_values)*cos(phi) - w_values*sin(phi))))
    I_ext_interpolator = interp1d(time_values, I_ext)
    return I_ext_interpolator


class ECT_current(df.Expression):
    """Expresion for forcing Neumann boundary condition."""

    def __init__(self, time, interval, amplitude, period, duration, area, **kwargs) -> None:
        """Create an interolated forcing function."""
        self.time = time
        self.area = area
        self.I_ext_callable = get_I_ext(interval, amplitude, period, duration)

    def eval(self, value, x) -> None:
        """Evaluate the interpolated function and scale by area."""
        value[0] = self.I_ext_callable(self.time(0))/self.area


def  _example1():
    interval = 4
    I_ext_interpolator = get_I_ext(interval, amplitude=800, period=0.5, duration=0.25)

    import matplotlib.pyplot as plt
    import seaborn as sns

    time = np.linspace(0, interval, 100*interval)
    values = I_ext_interpolator(time)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, values)

    ax.set_title("A square pulse forcing")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")

    fig.savefig("forcing.png")


if __name__ == "__main__":
    _example1()
