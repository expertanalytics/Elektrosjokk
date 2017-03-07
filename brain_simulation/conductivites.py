"""Overloaded Expressions representing the intra- and extracellular conductances."""

from dolfin import Expression


__all__ = ["IntracellularConductivity", "ExtracellularConductivity", "Water"]


class IntracellularConductivity(Expression):
    """Intracellular conductivity tensor."""

    def eval(self, value, x):
        """Overloaded eval method."""
        c = 3.71e3
        value[0] = c
        value[1] = 0.0
        value[2] = 0.0

        value[3] = 0.0
        value[4] = c
        value[5] = 0.0

        value[6] = 0.0
        value[7] = 0.0
        value[8] = c

    def value_shape(self):
        """Return the shape of `eval`."""
        return (3, 3)


class ExtracellularConductivity(Expression):
    """Extracellular conductivity tensor."""

    def eval(self, value, x):
        """Overloaded eval method."""
        c = 3.71e3
        value[0] = c
        value[1] = 0.0
        value[2] = 0.0

        value[3] = 0.0
        value[4] = c
        value[5] = 0.0

        value[6] = 0.0
        value[7] = 0.0
        value[8] = c

    def value_shape(self):
        """Return the shape of `eval`."""
        return (3, 3)


class Water(Expression):
    """Conductivity tensor for CSF."""

    def eval(self, value, x):
        """Overloaded eval method."""
        c = 1.79e3
        value[0] = c
        value[1] = 0.0
        value[2] = 0.0

        value[3] = 0.0
        value[4] = c
        value[5] = 0.0

        value[6] = 0.0
        value[7] = 0.0
        value[8] = c

    def value_shape(self):
        """Return the shape of `eval`."""
        return (3, 3)
