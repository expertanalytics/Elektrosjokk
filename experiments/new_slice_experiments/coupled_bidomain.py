import dolfin as df
import numpy as np

from typing import (
    Dict,
    Tuple,
    Any,
    Iterator,
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
    CoupledBidomainParameters,
    time_stepper,
)


class CoupledBidomainSolver:
    def __init__(
        self,
        time: df.Constant,
        mesh: df.Mesh,
        intracellular_conductivity: Dict[int, df.Expression],
        extracellular_conductivity: Dict[int, df.Expression],
        cell_function: df.MeshFunction,
        cell_tags: CellTags,
        interface_function: df.MeshFunction,
        interface_tags: InterfaceTags,
        parameters: CoupledBidomainParameters,
        neumann_boundary_condition: Dict[int, df.Expression] = None,
        v_prev: df.Function = None,
    ) -> None:
        self._time = time
        self._mesh = mesh

        if not set(intracellular_conductivity.keys()) == set(extracellular_conductivity.keys()):
            raise ValueError("intracellular conductivity and lambda does not have natching keys.")
        self._intracellular_conductivity = intracellular_conductivity
        self._extracellular_conductivity = extracellular_conductivity

        # TODO: check matching keys and tags etc.
        self._cell_function = cell_function
        self._cell_tags = cell_tags
        self._interface_function = interface_function
        self._interface_tags = interface_tags
        self._parameters = parameters

        # Set up function spaces
        self._transmembrane_function_space = df.FunctionSpace(self._mesh, "CG", 1)
        transmembrane_element = df.FiniteElement("CG", self._mesh.ufl_cell(), 1)
        extracellular_element = df.FiniteElement("CG", self._mesh.ufl_cell(), 1)

        if neumann_boundary_condition is None:
            self._neumann_bc: Dict[int, df.Expression] = dict()
        else:
            self._neumann_bc = neumann_boundary_condition

        if self._parameters.linear_solver_type == "direct":
            lagrange_element = df.FiniteElement("R", self._mesh.ufl_cell(), 0)
            mixed_element = df.MixedElement((transmembrane_element, extracellular_element, lagrange_element))
        else:
            mixed_element = df.MixedElement((transmembrane_element, extracellular_element))
        self._VUR = df.FunctionSpace(mesh, mixed_element)    # TODO: rename to something sensible

        # Set-up solution fields:
        if v_prev is None:
            self._merger = df.FunctionAssigner(self._transmembrane_function_space, self._VUR.sub(0))
            self._v_prev = df.Function(self._transmembrane_function_space, name="v_prev")
        else:
            self._merger = None
            self._v_prev = v_prev
        self._vur = df.Function(self._VUR, name="vur")        # TODO: Give sensible name

        # Mark first timestep
        self._timestep: df.Constant = None

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return self._v_prev, self._vur

    def _create_linear_solver(self):
        """Helper function for creating linear solver based on parameters."""
        solver_type = self._parameters.linear_solver_type

        if solver_type == "direct":
            solver = df.LUSolver(self._lhs_matrix)
        elif solver_type == "iterative":
            alg = self.parameters.krylov_method
            prec = self.parameters.krylov_preconditioner

            solver = df.PETScKrylovSolver(alg, prec)
            solver.set_operator(self._lhs_matrix)
            solver.parameters["nonzero_initial_guess"] = True
        else:
            msg = "Unknown solver type. Got {}, expected 'iterative' or 'direct'".format(solver_type)
            raise ValueError(msg)
        return solver

    def variational_forms(self, kn: df.Constant) -> Tuple[Any, Any]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          kn (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """
        # Extract theta parameter and conductivities
        theta = self._parameters.theta
        Mi = self._intracellular_conductivity
        Me = self._extracellular_conductivity

        # Define variational formulation
        if self._parameters.linear_solver_type == "direct":
            v, u, multiplier = df.TrialFunctions(self._VUR)
            v_test, u_test, multiplier_test = df.TestFunctions(self._VUR)
        else:
            v, u = df.TrialFunctions(self._VUR)
            v_test, u_test = df.TestFunctions(self._VUR)

        Dt_v = (v - self._v_prev)/kn
        v_mid = theta*v + (1.0 - theta)*self._v_prev

        # Set-up measure and rhs from stimulus
        dOmega = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_function)
        dGamma = df.Measure("ds", domain=self._mesh, subdomain_data=self._interface_function)

        # Loop over all domains
        G = Dt_v*v_test*dOmega()
        for key in set(self._cell_tags):        # TODO: do I need set here? cell_tags is a tuple. But is a guard against dicts
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(v_test))*dOmega(key)
            G += df.inner(Mi[key]*df.grad(u), df.grad(v_test))*dOmega(key)
            G += df.inner(Mi[key]*df.grad(v_mid), df.grad(u_test))*dOmega(key)
            G += df.inner((Mi[key] + Me[key])*df.grad(u), df.grad(u_test))*dOmega(key)

            # If Lagrangian multiplier
            if self._parameters.linear_solver_type == "direct":
                G += (multiplier_test*u + multiplier*u_test)*dOmega(key)

        for key in set(self._interface_tags):
            # Default to 0 if not defined for tag
            # I do not I should apply `chi` here.
            G += self._neumann_bc.get(key, df.Constant(0))*u_test*dGamma(key)

        a, L = df.system(G)
        return a, L

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None
    ) -> Iterator[Tuple[Tuple[float, float], Tuple[df.Function, df.Function]]]:
        """
        Solve the discretization on a given time interval (t0, t1)
        with a given timestep dt and return generator for a tuple of
        the interval and the current solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, solution_field) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, v = solution_fields
            # do something with the solutions
        """
        for interval in time_stepper(t0=t0, t1=t1, dt=dt):
            self.step(*interval)
            yield interval, self.solution_fields()

            # TODO: Update wlsewhere?
            self._v_prev.assign(self._vur.sub(0))

    def step(self, t0: float, t1: float) -> None:
        r"""
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """
        dt = t1 - t0
        theta = self._parameters.theta
        t = t0 + theta*dt
        self._time.assign(t)

        # Update matrix and linear solvers etc as needed
        if self._timestep is None:
            self._timestep = df.Constant(dt)
            self._lhs, self._rhs = self.variational_forms(self._timestep)

            # Preassemble left-hand side and initialize right-hand side vector
            self._lhs_matrix = df.assemble(self._lhs)
            self._rhs_vector = df.Vector(self._mesh.mpi_comm(), self._lhs_matrix.size(0))
            self._lhs_matrix.init_vector(self._rhs_vector, 0)

            # Create linear solver (based on parameter choices)
            self._linear_solver = self._create_linear_solver()
        else:
            self._update_solver(dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        print("FIXME: move extracellular indices somewhere else -- l 646 bidomainsolver")
        # TODO: This smells of dofmapping
        extracellular_indices = np.arange(0, self._rhs_vector.size(), 2)
        rhs_norm = self._rhs_vector.get_local()[extracellular_indices].sum()
        rhs_norm /= extracellular_indices.size

        # TODO: What is this?
        # rhs_norm = self._rhs_vector.array()[extracellular_indices].sum()/extracellular_indices.size
        self._rhs_vector.get_local()[extracellular_indices] -= rhs_norm

        # Solve problem
        self._linear_solver.solve(
            self._vur.vector(),
            self._rhs_vector
        )

    def _update_solver(self, dt: float) -> None:
        """Update the lhs matrix if timestep changes."""
        if (abs(dt - float(self._timestep)) < 1e-12):
            return
        self._timestep.assign(df.Constant(dt))

        # Reassemble matrix
        df.assemble(self._lhs, tensor=self._lhs_matrix)
