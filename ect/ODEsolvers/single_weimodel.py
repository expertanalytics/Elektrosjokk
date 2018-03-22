"""A Wrapper around xalbrain ODE solver used for solving cell models."""

import logging

from xalbrain import (
    SingleCellSolver,
    Constant,
    Parameters,
    Function,
)

from Typing import (
    Dict,
    Tuple,
)


logger = logging.getLogger(__name__)


def fenics_ode_solver(
        model: CardiacCellModel,
        ic: Dict[str, float] = None,
        interval: Tuple[float],
        dt: float,
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

    time = Constant(t0)

    vs_, vs = solver.solution_fields()
    model.set_initial_conditions(**ic)
    vs_.assign(model.initial_conditions())

    interval = (t0, t1)
    
    for (t0, t1), solution in solver.solve(interval, dt):
        yield (t0, t1), solution