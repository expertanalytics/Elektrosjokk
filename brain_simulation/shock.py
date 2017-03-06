from fenics import *


class Shock3D(Expression):
    def __init__(self, **kwargs):
        "Time dependent expression for external electrical stimulus."
        self.t = kwargs["t"]    # in ms
        self.amplitude = kwargs["amplitude"]    # mA
        self.spread = kwargs["spread"]
        self.x, self.y, self.z = kwargs["center"]   # mm
        self.half_period = kwargs["half_period"]          # ms

    def eval(self, value, x):
        A = self.amplitude
        s = self.spread
        if abs(sin(2*pi/self.half_period*float(self.t))) >= 0.5:      # Create a periodic square pulse
        #if self.t > 0.5:
            value[0] = A*exp(-s*pow(x[0] - self.x, 2))*exp(-s*pow(x[1] - self.y, 2))*\
            exp(-s*pow(x[2] - self.z, 2))
        else:
            value[0] = 0


def get_shock(t=0):
    S = Shock3D(t=t, spread=0.003, amplitude=8e1, center=(31, -15, 73), half_period=1e0, degree=1)
    return S
