from dolfin import *


mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

f = Constant(0)*v*dx
F = inner(grad(u), grad(v))*dx
