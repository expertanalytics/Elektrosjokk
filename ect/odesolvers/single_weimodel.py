"""A Wrapper around xalbrain ODE solver used for solving cell models."""

import logging

from xalbrain import (
    SingleCellSolver,
    Constant,
    Parameters,
    Function,
    CardiacCellModel,
)

from typing import (
    Dict,
    Tuple,
)


logger = logging.getLogger(__name__)


def fenics_ode_solver(
        model: CardiacCellModel,
        time: Constant,
        dt: float,
        interval: Tuple[float],
        ic: Dict[str, float] = None,
        params: Parameters = None
) -> Tuple[float, Function]:
    """Solve a single cell model in a specified time interval.

    The solutions are computed lazily, i.e. 

    ```
    for interval, solution in fenics_ode_solver( ... ):
        print(interval, solution)
    ```
    
    Args:
        model: An instance of `CardiacCellModel`.
        ic: A Dictionary of {name: value} pairs. The name must match a variable in the 
            cell model.
        interval: Solve the ODE in the interval (t0, t1).
        dt: Time step.
        params: Parameters for `SingleCellSolver`.
    """
    if params is None:
        params = SingleCellSolver.default_parameters()

    if ic is None:
        ic = model.initial_conditions()

    assert time(0) == interval[0]

    solver = SingleCellSolver(model, time, params)
    vs_, vs = solver.solution_fields()
    model.set_initial_conditions(**ic)
    vs_.assign(model.initial_conditions())
    
    for (t0, t1), solution in solver.solve(interval, dt):
        yield t1, solution.vector().get_local()[:model.num_states() + 1]
