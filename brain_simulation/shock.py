from dolfin import Expression, sin, exp, pi


__all__ = ["Shock3D"]


class Shock3D(Expression):
    """Time dependent expression for external electrical stimulus."""

    def __init__(self, **kwargs):
        """Constructor using only **kwargs."""
        self.t = kwargs["t"]    # in ms
        self.amplitude = kwargs["amplitude"]    # mA
        self.spread = kwargs["spread"]
        self.x, self.y, self.z = kwargs["center"]   # mm
        self.half_period = kwargs["half_period"]          # ms

    def eval(self, value, x):
        """Overloaded eval method."""
        A = self.amplitude
        s = self.spread

        # Create a periodic square pulse
        if abs(sin(2*pi/self.half_period*float(self.t))) >= 0.5:
            value[0] = A*exp(-s*pow(x[0] - self.x, 2))*exp(-s*pow(x[1] - self.y, 2))*\
                       exp(-s*pow(x[2] - self.z, 2))
        else:
            value[0] = 0
