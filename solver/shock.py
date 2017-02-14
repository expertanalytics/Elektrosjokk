from fenics import *


class shock3D(Expression):
    def __init__(self, **kwargs):
        "Time dependent expression for external electrical stimulus."
        self.t = kwargs["t"]
        self.amplitude = kwargs["amplitude"]
        self.x, self.y, self.z = kwargs["center"]
        self.period = kwargs["period"]

    def eval(self, value, x): 
        A = self.amplitude
        if sin(2*pi/self.period*self.t) >= 0.5:      # Create a periodic square pulse
            value[0] = 100*exp(A*pow(x[0] - self.x, 2))*exp(A*pow(x[1] - self.y, 2))*\
            exp(A*pow(x[2] - self.z))
        else:
            value[0] = 0


def get_shock(t=0):
    S = shock3D(t=t, amplitude=-0.003, center=(-34.6, 0.28, -87.0), period=1e0, degree=1)
    return S
