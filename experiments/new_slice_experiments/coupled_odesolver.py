import dolfin as df

from bbidomain import VectorInt

from extension_modules import load_module

from typing import (
    Tuple,
    Union,
    Dict,
    NamedTuple,
    Sequence,
    Iterator,
)

from utils import (
    masked_dofs,
    time_stepper,
    CoupledODESolverParameters,
)

from xalbrain.cellmodels import CardiacCellModel


class CoupledODESolver:
    def __init__(
            self,
            time: df.Constant,
            mesh: df.Mesh,
            model: CardiacCellModel,
            parameters: CoupledODESolverParameters,
            cell_function: df.MeshFunction = None,
    ) -> None:
        """Initialise parameters. NB! Keep I_s for compatibility"""
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = model     # FIXME: For initial conditions and num states

        # Extract some information from cell model
        self._num_states = self._model.num_states()

        self._parameters = parameters
        valid_cell_tags = self._parameters.valid_cell_tags

        # Create (vector) function space for potential + states
        self._function_space_VS = df.VectorFunctionSpace(self._mesh, "CG", 1, dim=self._num_states + 1)

        # Initialize solution field
        self.vs_prev = df.Function(self._function_space_VS, name="vs_prev")
        self.vs = df.Function(self._function_space_VS, name="vs")

        dofmaps = [
            self._function_space_VS.sub(i).dofmap() for i in range(self._function_space_VS.num_sub_spaces())
        ]

        if valid_cell_tags is None or cell_function is None:
            self._dofs = [VectorInt(dofmap.dofs() for dofmap in dofmaps)]
        else:
            self._dofs = [
                masked_dofs(dofmap, cell_function.array(), valid_cell_tags) for dofmap in dofmaps
            ]

        model_name = model.__class__.__name__        # Which module to load
        self.ode_module = load_module(
            model_name,
            recompile=self._parameters.reload_ext_modules,
            verbose=self._parameters.reload_ext_modules
        )
        self.ode_solver = self.ode_module.BetterODESolver(*self._dofs)

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.
        """
        return self.vs_prev, self.vs

    def step(self, t0: float, t1: float) -> None:
        """Take a step using my much better ode solver."""
        dt = t1 - t0        # TODO: Is this risky?
        self.ode_solver.solve(self.vs_prev.vector(), t0, t1, dt)
        self.vs.assign(self.vs_prev)

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None,
    ) -> Iterator[Tuple[Tuple[float, float], df.Function]]:
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Example of usage*::

          # Create generator
          solutions = solver.solve(0.0, 1.0, 0.1)

          # Iterate over generator (computes solutions as you go)
          for interval, vs in solutions:
            # do something with the solutions

        """
        # Solve on entire interval if no interval is given.
        for t0, t1 in time_stepper(t0=t0, t1=t1, dt=dt):
            self.step(t0, t1)

            # Yield solutions
            yield (t0, t1), self.vs
            self.vs_prev.assign(self.vs)


class BetterSingleCellSolver(BetterODESolver):
    def __init__(
            self,
            model: CardiacCellModel,
            time: df.Constant,
            reload_ext_modules: bool = False,
            params: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        assert isinstance(model, CardiacCellModel), \
            "Expecting model to be a CardiacCellModel, not %r" % model
        assert (isinstance(time, df.Constant)), \
            "Expecting time to be a Constant instance, not %r" % time
        assert isinstance(params, df.Parameters) or params is None, \
            "Expecting params to be a Parameters (or None), not %r" % params

        # Store model
        self._model = model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(
            mesh,
            time,
            model,
            I_s=model.stimulus,
            reload_ext_modules=reload_ext_modules,
            params=params
        )
