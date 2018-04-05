"""A collection of tools to assign inhomogeneous initial conditions."""

import numpy as np
import xalbrain as xb

from typing import (
    Callable,
    Tuple,
)


class NonuniformIC:
    def __init__(self, coordinates: np.ndarray, data: np.ndarray) -> None:
        """
        Arguments:
            coordinates: Original sampling coordinates for `data`.
            data: Samples values at `coordinates`.
        """
        self.data = data
        cmax = coordinates.max()
        cmin = coordinates.min()
        self.coordinates = (cmax - cmin)*np.linspace(0, 1, data.shape[0]) + cmin

    def __call__(self) -> np.ndarray:
        for i in range(self.data.shape[1]):
            yield lambda x: np.interp(x, self.coordinates, self.data[:, i])


def new_assign_ic(
        receiving_function: xb.Function,
        ic_generator: NonuniformIC,
        degree: int = 1
) -> None:
    """
    Assign receiving_function(x, y) <- `ic_function`(x, y), for x, in the mesh.

    Arguments:
        receiving_function: The function which is assigned the initial condition.
        ic_function_tuple: A tuple of python callables which return the initial condition for each
            point (x, y). The number of functions must match the number of subfunctions 
            in `receiving_function`.
    """
    mixed_func_space = receiving_function.function_space()
    mesh = mixed_func_space.mesh()
    V = xb.FunctionSpace(mesh, "CG", 1)    # TODO: infer this somehow

    # Copy functions to be able to assign to them
    functions = receiving_function.split(deepcopy=True)

    for i, (f, ic_func) in enumerate(zip(functions, ic_generator())):
        class IC(xb.Expression):
            def eval(self, value, x):
                value[0] = ic_func(x[0])    # TODO: 1D for now

        ic = IC(degree=degree)
        assigner = xb.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(receiving_function.sub(i), xb.project(ic, V))


def assign_ic(func):
    mixed_func_space = func.function_space()

    functions = func.split(deepcopy=True)
    ic = get_random_time(functions[0].vector().size())
    V = xb.FunctionSpace(mixed_func_space.mesh(), "CG", 1)

    for i, f in enumerate(functions):
        ic_func = xb.Function(V)
        ic_func.vector()[:] = np.array(ic[:, i])

        assigner = xb.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(func.sub(i), ic_func)
