from dolfin import Expression


class IntracellularConductivity(Expression):
    "Intracellular conductivity tensor."
    def eval(self, value, x):
        value[0] = 100.0
        value[1] = 0.0
        value[2] = 0.0

        value[3] = 0.0
        value[4] = 100.0
        value[5] = 0.0

        value[6] = 0.0
        value[7] = 0.0
        value[8] = 100.0

    def value_shape(self):
      return (3, 3)


class ExtracellularConductivity(Expression):
    "Extracellular conductivity tensor."
    def eval(self, value, x):
        value[0] = 10.0
        value[1] = 0.0
        value[2] = 0.0

        value[3] = 0.0
        value[4] = 10.0
        value[5] = 0.0

        value[6] = 0.0
        value[7] = 0.0
        value[8] = 10.0

    def value_shape(self):
        return (3, 3)


class Water(Expression):
    "Conductivity tensor for CSF."

    def eval(self, value, x):
        value[0] = 3.0
        value[1] = 0.0
        value[2] = 0.0

        value[3] = 0.0
        value[4] = 3.0
        value[5] = 0.0

        value[6] = 0.0
        value[7] = 0.0
        value[8] = 3.0

    def value_shape(self):
        return (3, 3)
