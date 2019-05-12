import dolfin as df

from coupled_utils import (
    time_stepper,
    CoupledSplittingsolverParameters,
    CoupledMonodomainParameters,
    CoupledODESolverParameters
)

from typing import (
    Iterator,
    Tuple,
)

from coupled_odesolver import CoupledODESolver

from coupled_monodomain import (
    CoupledMonodomainSolver
)

from coupled_brainmodel import CoupledBrainModel


class CoupledSplittingsolver:
    def __init__(
            self,
            brain: CoupledBrainModel,
            parameters: CoupledSplittingsolverParameters,
    ):
        """Create solver from given Cardiac Model and (optional) parameters."""
        self._brain = brain
        self._parameters = parameters

        # Create ODE solver and extract solution fields
        self.ode_solver = self.create_ode_solver()
        self.vs_, self.vs = self.ode_solver.solution_fields()
        self.VS = self.vs.function_space()

        # Create PDE solver and extract solution fields
        self.pde_solver = self._create_pde_solver()
        self.v_, self.vur = self.pde_solver.solution_fields()

        # # Create function assigner for merging v from self.vur into self.vs[0]
        if self.VS.num_sub_space() == 2:
            V = self.vur.function_space().sub(0)
        elif self.VS.num_sub_space() == 0:
            V = self.vur.function_space()
        else:
            raise TypeError("Expected function space with 0 or two sub spaces.")
        self.merger = df.FunctionAssigner(self.VS.sub(0), V)

    def create_ode_solver(self) -> CoupledODESolver:
        """The idea is to subplacc this and implement another version of this function."""
        parameters = CoupledODESolverParameters()
        solver = CoupledODESolver(
            self.brain.time,
            self.brain.mesh,
            self.brain.cell_model,
            parameters,
            self.brain.cell_function
        )
        return solver

    def create_pde_solver(self) -> CoupledMonodomainSolver:
        """The idea is to subplacc this and implement another version of this function."""
        parameters = CoupledMonodomainParameters()
        solver = CoupledMonodomainSolver(
            self._brain.time,
            self._brain.mesh,
            self._brain.intracellular_conductivity,
            self._brain.extracellular_conductivity,
            self._brain.cell_function,
            self._brain.cell_tags,
            self._brain.interface_function,
            self._brain.interface_tags,
            parameters,
            self._brain.neumann_boundary_conditions
        )
        return solver

    def merge(self, solution: df.Function) -> None:
        """
        Combine solutions from the PDE and the ODE to form a single mixed function.

        `solution` holds the solution from the PDEsolver.
        """
        if self.VS.num_sub_space() == 2:
            v = self.vur.sub(0)
        else:
            v = self.vur
        self.merger.assign(solution.sub(0), v)

    def solve(self, t0: float, t1: float, dt: float) -> Iterator[Tuple[Tuple[float, float], df.Function]]:
        """
        Solve the problem given by the model on a time interval with a given time step.
        Return a generator for a tuple of the time step and the solution fields.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, list of tuples of floats)
            The timestep for the solve. A list of tuples of floats can
            also be passed. Each tuple should contain two floats where the
            first includes the start time and the second the dt.

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          dts = [(0., 0.1), (1.0, 0.05), (2.0, 0.1)]
          solutions = solver.solve((0.0, 1.0), dts)

          # Iterate over generator (computes solutions as you go)
          for ((t0, t1), (vs_, vs, vur)) in solutions:
            # do something with the solutions

        """
        # Create timestepper

        for t0, t1 in time_stepper(t0=t0, t1=t1, dt=dt):
            self.step(t0, t1, dt)

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Update previous solution
            self.vs_.assign(self.vs)

    def step(self, t0: float, t1: float) -> None:
        """
        Solve the pde for one time step.

        Invariants:
            Given self._vs in a correct state at t0, provide v and s (in self.vs) and u
            (in self.vur) in a correct state at t1.
            (Note that self.vur[0] == self.vs[0] only if theta = 1.0.)
        """
        theta = self._parameters.theta

        # Extract time domain
        _dt = t1 - t0
        t = t0 + theta*_dt

        # Compute tentative membrane potential and state (vs_star)
        # df.begin(df.PROGRESS, "Tentative ODE step")
        # Assumes that its vs_ is in the correct state, gives its vs
        # in the current state
        # self.ode_solver.step((t0, t))
        self.ode_solver.step(t0, t)
        # print("ODE time: ", tock - tick)

        # self.vs_.assign(self.vs)

        # Compute tentative potentials vu = (v, u)
        # Assumes that its vs_ is in the correct state, gives vur in
        # the current state
        self.pde_solver.step(t0, t1)
        # print("PDE time: ", tock - tick)

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and the s part of its
            # vs are in the correct state, provides input argument(in
            # this case self.vs) in its correct state
            self.merge(self.vs)
            return

        # Otherwise, we do another ode_step:

        # Assumes that the v part of its vur and the s part of its vs
        # are in the correct state, provides input argument (in this
        # case self.vs_) in its correct state
        self.merge(self.vs_)    # self.vs_.sub(0) <- self.vur.sub(0)
        # Assumes that its vs_ is in the correct state, provides vs in the correct state

        # self.ode_solver.step((t0, t))
        self.ode_solver.step(t0, t)
