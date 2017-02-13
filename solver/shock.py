from fenics import *


class shock3D(Expression): 
    def __init__(self, **kwargs): 
        "Time dependent expression for external electrical stimulus."
        self.t = kwargs["t"]

    def eval(self, value, x): 
        if self.t <= 0.1: 
            value[0] = exp(-100*pow(x[0] - 34.6, 2))*exp(-100*pow(x[1] - 0.28, 2))*
            exp(-100*pow(x[2] - 87.0))
        else: 
            value[0] = 0     


def get_shock(t=0): 
    #S = shock3D(t=t, degree=1) 
    #S = Expression("exp(-100*pow(x[0]-34.6, 2))*exp(-100*pow(x[1]-0.28, 2))*exp(-100*pow(x[2]-87.0, 2))*exp(-t)", t=t) 
    #S = Expression("x[0]-34.6", degree=1) 
    S = Expression("exp(-0.003*pow(x[0] - 34.6, 2))*exp(-0.003*pow(x[1] - 0.28, 2))\
                   *exp(-0.003*pow(x[2] - 87.0, 2))*exp(-t)", t=t, degree=1) 
    return S  
