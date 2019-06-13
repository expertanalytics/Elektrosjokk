import dolfin as df

from typing import (
    Dict,
    Any,
    Tuple,
    NamedTuple,
    Iterable,
    Union,
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
    CoupledMonodomainParameters,
    create_linear_solver,
    time_stepper,
)

from IPython import embed


class CoupledMonodomainSolver:
    def __init__(
        self,
        time: df.Constant,
        mesh: df.Mesh,
        conductivity: Dict[int, df.Expression],
        conductivity_ratio: Dict[int, df.Expression],
        cell_function: df.MeshFunction,
        cell_tags: CellTags,
        interface_function: df.MeshFunction,
        interface_tags: InterfaceTags,
        parameters: CoupledMonodomainParameters,
        neumann_boundary_condition: Dict[int, df.Expression] = None,
        external_stimulus: Dict[int, df.Expression] = None,
        v_prev: df.Function = None
    ) -> None:
        self._time = time
        self._mesh = mesh
        self._conductivity = conductivity
        self._parameters = parameters
        self._cell_function = cell_function
        self._cell_tags = cell_tags
        self._interface_function = interface_function
        self._interface_tags = interface_tags

        if neumann_boundary_condition is None:
            self._neumann_boundary_condition: Dict[int, df.Expression] = dict()
        else:
            self._neumann_boundary_condition = neumann_boundary_condition

        if external_stimulus is None:
            self._external_stimulus: Dict[int, df.Expression] = dict()
        else:
            self._external_stimulus = external_stimulus

        if not set(conductivity.keys()) == set(conductivity_ratio.keys()):
            raise ValueError("intracellular conductivity and lambda does not have natching keys.")
        self._lambda = conductivity_ratio

        # Function spaces
        self._function_space = df.FunctionSpace(mesh, "CG", 1)

        # Test and trial and previous functions
        self._v_trial = df.TrialFunction(self._function_space)
        self._v_test = df.TestFunction(self._function_space)

        self._v = df.Function(self._function_space)
        if v_prev is None:
            self._v_prev = df.Function(self._function_space)
        else:
            # v_prev is shipped from an odesolver.
            self._v_prev = v_prev

        _cell_tags = set(self._cell_tags)
        _cell_function_values = set(self._cell_function.array())
        if not _cell_tags == _cell_function_values:
            msg = f"Cell function does not contain {_cell_tags - _cell_function_values}"
            raise ValueError(msg)

        _interface_tags = set(self._interface_tags)
        _interface_function_values = {*set(self._interface_function.array()), None}
        if not _interface_tags <= _interface_function_values:
            msg = f"interface function does not contain {_interface_tags - _interface_function_values}."
            raise ValueError(msg)

        # Crete integration measures -- Interfaces
        self._dGamma = df.Measure("ds", domain=self._mesh, subdomain_data=self._interface_function)

        # Crete integration measures -- Cells
        self._dOmega = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_function)

        # Create variational forms
        self._timestep = df.Constant(self._parameters.timestep)
        self._lhs, self._rhs = self._variational_forms()

        # Preassemble left-hand side (will be updated if time-step changes)
        # self._lhs_matrix, self._rhs_vector = df.assemble_system(self._lhs, self._rhs, [])

        self._lhs_matrix = df.assemble(self._lhs)
        self._rhs_vector = df.Vector(mesh.mpi_comm(), self._lhs_matrix.size(0))
        self._lhs_matrix.init_vector(self._rhs_vector, 0)

        self._linear_solver = create_linear_solver(self._lhs_matrix, self._parameters)

    def _variational_forms(self) -> Tuple[Any, Any]:
        # Localise variables for convenicence
        dt = self._timestep
        dt = 0.025
        theta = self._parameters.theta
        Mi = self._conductivity
        lbda = self._lambda

        dOmega = self._dOmega
        dGamma = self._dGamma

        v = self._v_trial
        v_test = self._v_test

        # Set-up variational problem
        dvdt = (v - self._v_prev)/dt
        v_mid = theta*v + (1.0 - theta)*self._v_prev

        # Cell contributions
        Form = dvdt*v_test*dOmega()
        for cell_tag in self._cell_tags:
            Form += lbda[cell_tag]*df.inner(Mi[cell_tag]*df.grad(v_mid), df.grad(v_test))*dOmega(cell_tag)
            Form += self._external_stimulus.get(cell_tag, df.Constant(0))*v_test*dOmega(cell_tag)

        # Boundary contributions
        for interface_tag in self._interface_tags:
            neumann_bc = self._neumann_boundary_condition.get(interface_tag, df.Constant(0))
            Form += neumann_bc*v_test*dGamma(interface_tag)

        a, L = df.system(Form)
        return a, L

    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """Return current and previous solution."""
        return self._v_prev, self._v

    def step(self, t0, t1) -> None:
        # Extract interval and thus time-step
        theta = self._parameters.theta
        dt = t1 - t0
        t = t0 + theta*dt
        self._time.assign(t)

        # Update matrix and linear solvers etc as needed
        self._update_solver(dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        # Solve problem
        self._linear_solver.solve(
            self._v.vector(),
            self._rhs_vector
        )

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None
    ) -> Iterable[Tuple[Tuple[float, float], Tuple[df.Function, df.Function]]]:
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
            # info("Solving on t = (%g, %g)" % (t0, t1))
            self.step(interval)

            # Yield solutions
            yield interval, self.solution_fields()

            # Update wlsewhere???
            self._v_prev.assign(self._v)

    def _update_solver(self, dt: float) -> None:
        """Update the lhs matrix if timestep changes."""
        if (abs(dt - float(self._timestep)) < 1e-12):
            return
        self._timestep.assign(df.Constant(dt))

        assert False, "This probably breaks the sparsity pattern"
        # Reassemble matrix
        df.assemble(self._lhs, tensor=self._lhs_matrix)


class NetworkMonodomainSolver(CoupledMonodomainSolver):
    def __init__(
        self,
        time: df.Constant,
        mesh: df.Mesh,
        conductivity: Dict[int, df.Expression],
        conductivity_ratio: Dict[int, df.Expression],
        cell_function: df.MeshFunction,
        cell_tags: CellTags,
        interface_function: df.MeshFunction,
        interface_tags: InterfaceTags,
        parameters: CoupledMonodomainParameters,
        neumann_boundary_condition: Dict[int, df.Expression] = None,
        external_stimulus: Dict[int, df.Expression] = None,
        v_prev: df.Function = None,
    ) -> None:
        self._time = time
        self._mesh = mesh
        self._conductivity = conductivity
        self._cell_function = cell_function
        self._cell_tags = cell_tags
        self._interface_function = interface_function
        self._interface_tags = interface_tags
        self._parameters = parameters

        if neumann_boundary_condition is None:
            self._neumann_boundary_condition: Dict[int, df.Expression] = dict()
        else:
            self._neumann_boundary_condition = neumann_boundary_condition

        if external_stimulus is None:
            self._external_stimulus: Dict[int, df.Expression] = dict()
        else:
            self._external_stimulus = external_stimulus

        if not set(conductivity.keys()) == set(conductivity_ratio.keys()):
            raise ValueError("intracellular conductivity and lambda does not have natching keys.")
        self._lambda = conductivity_ratio

        # Function spaces
        self._function_space = df.FunctionSpace(mesh, "CG", 1)

        # Test and trial and previous functions
        self._v_trial = df.TrialFunction(self._function_space)
        self._v_test = df.TestFunction(self._function_space)

        self._v = df.Function(self._function_space)
        if v_prev is None:
            self._v_prev = df.Function(self._function_space)
        else:
            # v_prev is shipped from an odesolver.
            self._v_prev = v_prev

        _cell_tags = set(self._cell_tags)
        _cell_function_values = set(self._cell_function.array())
        if not _cell_tags == _cell_function_values:
            msg = f"Cell function does not contain {_cell_tags - _cell_function_values}"
            raise ValueError(msg)

        _interface_tags = set(self._interface_tags)
        _interface_function_values = {*set(self._interface_function.array()), None}
        if not _interface_tags <= _interface_function_values:
            msg = f"interface function does not contain {_interface_tags - _interface_function_values}."
            raise ValueError(msg)

        # Crete integration measures -- Interfaces
        self._dGamma = df.Measure("ds", domain=self._mesh, subdomain_data=self._interface_function)

        # Crete integration measures -- Cells
        self._dOmega = df.Measure("dx", domain=self._mesh, subdomain_data=self._cell_function)
        self._variational_forms()

        self._linear_solver = create_linear_solver(self._lhs_matrix, self._parameters)

    def _variational_forms(self) -> None:
        # Localise variables for convenicence
        dt = df.Constant(self._parameters.timestep)
        self._timestep = dt
        theta = self._parameters.theta
        Mi = self._conductivity
        lbda = self._lambda

        dOmega = self._dOmega
        dGamma = self._dGamma

        v = self._v_trial
        v_test = self._v_test

        # Set-up variational problem
        dvdt = (v - self._v_prev)/dt
        v_mid = theta*v + (1.0 - theta)*self._v_prev

        # Cell contributions
        # Form = dvdt*v_test*domega()
        mass = dvdt*v_test*dOmega()

        stiffness = 0
        for cell_tag in self._cell_tags:
            stiffness += lbda[cell_tag]*df.inner(Mi[cell_tag]*df.grad(v_mid), df.grad(v_test))*dOmega(cell_tag)
            stiffness += self._external_stimulus.get(cell_tag, df.Constant(0))*v_test*dOmega(cell_tag)

        # Boundary contributions -- Harmonic BCs are hacked in
        a_mass, L_mass = df.system(mass)
        a_stiff, L_stiff = df.system(stiffness)
        self._rhs = L_mass + L_stiff
        self._rhs_vector = df.assemble(self._rhs)

        # lhs_mass = self._periodic_bc(df.assemble(a_mass))
        # lhs_stiff = self._periodic_bc(df.assemble(a_stiff))
        lhs_mass = df.as_backend_type(df.assemble(a_mass))
        lhs_stiff = df.assemble(a_stiff)
        import numpy as np
        import scipy.sparse as sp
        from petsc4py import PETSc
        np.set_printoptions(precision=3)

        ##################################################################################
        ##################################################################################

        full_stiff = lhs_stiff.array()
        N = full_stiff.shape[0]
        INDICES = np.arange(1, N - 1)

        RHO = int(0.01*N)    # percentage of refirings
        i = np.random.choice(INDICES, size=RHO)
        j = np.random.choice(INDICES[~np.in1d(INDICES, i)], size=i.size)
        assert not np.in1d(j, i).all() or RHO == 0

        # i = np.array([100])
        # j = i * 2 + 300

        # WEIGHT = full_stiff[0, 1]      # Off diagonal
        WEIGHT = 0.0001

        # full_stiff[i + 1, i] = 0
        # full_stiff[i - 1, i] = 0

        full_stiff[i, j] = WEIGHT
        full_stiff[j, i] = WEIGHT

        sparse_matrix = sp.csc_matrix(full_stiff.shape)
        sparse_matrix[:, :] = full_stiff

        p1 = sparse_matrix.indptr
        p2 = sparse_matrix.indices
        p3 = sparse_matrix.data
        lhs_stiff = df.PETScMatrix(PETSc.Mat().createAIJ(size=full_stiff.shape, csr=(p1, p2, p3)))

        # print(lhs_stiff.array())
        # print(lhs_mass.array())
        # embed()

        ##################################################################################
        ##################################################################################

        lhs_matrix = lhs_mass.mat() + lhs_stiff.mat()
        self._lhs_matrix = df.Matrix(df.PETScMatrix(lhs_matrix))
        # assert False

    def step(self, t0, t1) -> None:
        # Extract interval and thus time-step
        theta = self._parameters.theta
        dt = t1 - t0
        t = t0 + theta*dt
        self._time.assign(t)

        # Update matrix and linear solvers etc as needed
        self._update_solver(dt)

        # Assemble right-hand-side
        df.assemble(self._rhs, tensor=self._rhs_vector)

        # Super hacky -- Get a finite difference approximation
        # vector_size = self._rhs_vector.size()
        # self._rhs_vector[0] = self._rhs_vector[1]
        # self._rhs_vector[vector_size - 1] = self._rhs_vector[vector_size - 2]

        # Solve problem
        self._linear_solver.solve(
            self._v.vector(),
            self._rhs_vector
        )

    def _periodic_bc(self, petsc_matrix) -> Any:
        import scipy.sparse as sp
        from petsc4py import PETSc

        full_matrix = petsc_matrix.array()
        full_matrix[0, -1] = full_matrix[1, 0]
        full_matrix[0, 0] = full_matrix[1, 1]
        full_matrix[0, 1] = full_matrix[1, 2]

        full_matrix[-1, -2] = full_matrix[-2, -3]
        full_matrix[-1, -1] = full_matrix[-2, -2]
        full_matrix[-1, 0] = full_matrix[-2, -1]

        full_matrix[0, -1] = full_matrix[0, 1]
        full_matrix[-1, 0] = full_matrix[-1, -2]

        sparse_matrix = sp.csc_matrix(full_matrix.shape)
        sparse_matrix[:, :] = full_matrix

        p1 = sparse_matrix.indptr
        p2 = sparse_matrix.indices
        p3 = sparse_matrix.data
        petscmat = PETSc.Mat().createAIJ(size=full_matrix.shape, csr=(p1, p2, p3))
        return df.PETScMatrix(petscmat)
