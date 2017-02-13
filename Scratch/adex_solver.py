class AdEx_solver:
    def __init__(self, problem, dt, T):
        """Solver class for the AdEx model
        """
        assert isinstance(problem, AdEx_problem)
        self.dt = dt
        self.problem = problem
        self.T = T

    def solve(self, solver="cg", preconditioner="amg", family="CG", degree=1):
        """Variational form, matrix assembly and time loop

        Parameters
        ----------
        family : str, optional
            Finite element family (default 'CG')
        degree : int, optional
            Finite element polynomial degree (default is 1)
        solver : str, optional
            krylov solver method (default is 'cg')
        preconditioner : str, optional
            preconditioner (default is 'amg')
        """
        params = self.problem.params
        dtc = Constant(self.dt)

        I = self.problem.ionic_current()   # FIXME: What should this be?

        V = FunctionSpace(self.problem.mesh, family, degree)
        W = MixedFunctionSpace([V, V])

        v, w = TrialFunctions(W)
        p, q = TestFunctions(W)

        # Function for prefious time step
        vwp = Function(W)
        vp, wp = split(vwp)

        # Membrane potential equation
        F = params.C*(v - vp)/dtc*p*dx
        F += params.gl*(v - params.El)*p*dx
        F -= params.gl*params.Delta_t*exp((vp - params.El)/params.Delta_t)*p*dx     # linearise exp
        F += w*p*dx
        F += I*p*dx

        # Adaptation equation
        F += params.tau_w*(w - wp)/dtc*q*dx
        F -= params.a*(v - params.El)*q*dx
        F += w*q*dx

        L = rhs(F)
        a = lhs(F)

        A = assemble(a)
        b = assemble(L)

        # Set up solver
        solver = KrylovSolver(solver, preconditioner)     # TODO: Look into solver/preconditioner 
        solver.set_operator(A)

        vwe = Function(W)  # Store solution

        t = 0
        while t < self.T:    # Time loop
            I.t = t
            solver.solve(vwe.vector(), b)
            ve, we = split(vwe)

            # Rotate functions
            vwp.assign(vwe)

            # Update time
            t += self.dt

