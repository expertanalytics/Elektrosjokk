from dolfin import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

u = TestFunction(V)
v = TrialFunction(V)

bc = DirichletBC(V, Constant(0), DomainBoundary())

f = Expression("sin(2*pi*x[0])", degree=1)

a = inner(grad(u), grad(v))*dx 
b = f*v*dx

A, bb = assemble_system(a, b, [bc])

U_ = Function(V)

solver = PETScKrylovSolver("cg", "amg")
solver.set_operator(A)
solver.solve(U_.vector(), bb)

print(U_.vector().norm("l2"))
