from dolfin import *
from collections import namedtuple

class AdEx_problem:
    def __init__(self, mesh, **kwargs):
        """Container of AdEx related parameters

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

        for key in defaultparams:
            defaultparams[key] = Constant(params[key])

        self.mesh = mesh
        self.params = Parameters(**default_params)  # parameters subscriptable by name

    def initial_conditions(self):
        """Return problem specific initial conditions.
        """
        pass

    def boundary_conditions(self):
        """Return problem specific boundary conditions.
        """
        pass

    def update(self):
        """Update boundary conditions in time.
        """
        pass

    def ionic_current(self):
        """The input current for the AdEx model.

        Returns
        -------
        dolfin.Expression
            Time dependent expression for the input current
        """
        return Expression("1", t=0, degree=1)
