from dolfin import *
import numpy as np

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

# expr = Expression("x[0] + x[1]", "x[0] - x[1]", degree=1)
expr = Expression("x[0] + x[1]", degree=1)
u = project(expr, V)

data = u.vector().array()
data[data != 0] /= np.abs(data)

u.vector()[:] = data


for e in u.vector().array():
    print(f"{e:1.2f}")
