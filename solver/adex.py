from dolfin import *
from collections import namedtuple

class AdEx_problem:
    def __init__(self, mesh, **kwargs):
        """ Container of AdEx related parameters

        Parameters
        ----------
        mesh : dolin.Mesh
        C : float, optional
            Membrane capacitance (default is 281 pF)
        gl : float, optional
            Leak conductance (default is 30 ns)
        El : float, optional
            Leak reversal potential (default is -70.6 mV)
        Vt : float, optional
            Spike threshold (default is -50.4 mV)
        Delta_t : float, optional
            Slope factor (default is 2 mV)
        tau_w : float, optional
            Adaptation time constant (default is 144 ns)
        a : flaot, optional
            Subthreshold adaptation (default is 4 ns)
        b : Spike-triggered adaptation (default is 0.085 nA)
        """

        Parameters = namedtuple("Parameters", ["C", "a", "b", "gl", "El", "Delta_t", "Vt", "tau_w"])
        default_params = {"C" : 281,         # Membrane capacitance (pF)
                          "gl" : 30,         # Leak conductance (ns) 
                          "El" : -70.6,      # Leak reversal potential (mV)
                          "Vt" : -50.4,      # Spike threshold (mV)
                          "Delta_t" : 2,     # Slop factor (mV)
                          "tau_w" : 144,     # Adaptation time constant (ms)
                          "a" : 4,           # Subthreshold adaptation (nS)
                          "b" : 0.085        # Spike-triggered adaptation (nA)  
                          }

        # Make sure input parameters only has keys in default params 
        assert set(kwargs.keys()) <= set(default_params)

        for key in params:
            params[key] = Constant(params[key])

        self.mesh = mesh
        self.params = Parameters(**default_params)  # parameters subscriptable by name

    def initial_conditions(self):
        """ Return problem specific initial conditions
        """
        pass

    def boundary_conditions(self):
        """ Return problem specific boundary conditions
        """
        pass

    def update(self):
        """ Update boundary conditions in time
        """
        pass

    def ionic_current(self):
        """ The input current for the AdEx model

        Returns
        -------
        dolfin.Expression
            Time dependent expression for the input current
        """
        return Expression("1", t=0)

class AdEx_solver:
    def __init__(self, problem, dt, T):
        """ Solver class for the AdEx model
        """
        assert isinstance(problem, AdEx_problem)
        self.dt = dt
        self.problem = problem
        self.T = T

    def solve(self, solver="cg", preconditioner="amg", family="CG", degree=1):
        """ Variational form, matrix assembly and time loop

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
        W = V*V

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


if __name__ == "__main__":
    mesh = UnitIntervalMesh(10)

    params = {"C" : 281e-12,        # Membrane capacitance (pF)
              "gl" : 30e-9,         # Leak conductance (ns) 
              "El" : -70.6e-3,      # Leak reversal potential (mV)
              "Vt" : -50.4e-3,      # Spike threshold (mV)
              "Delta_t" : 2e-3,     # Slop factor (mV)
              "tau_w" : 144e-3,     # Adaptation time constant (ms)
              "a" : 4e-9,           # Subthreshold adaptation (nS)
              "b" : 0.085e-9        # Spike-triggered adaptation (nA)  
              }

    neuron = AdEx_problem(mesh)     # using default params
    neuron_solver = AdEx_solver(neuron, 1e-2, 1)
    neuron_solver.solve()
