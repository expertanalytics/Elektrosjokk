from dolfin import *

mesh = UnitSquareMesh(10, 10)

V = VectorFunctionSpace(mesh,  "CG", 1, dim=2)
v = Function(V)


# ic_expr = Expression("(u, v)", u=1, v=-1, degree=1)
# ic_expr = Expression(("u : uf ? x[0] > 0.5", "v"), u=1, v=-1, uf=-2, degree=1)
ic_expr = Expression(("x[0] > 0.5 ? u : uf", "v"), u=1, v=-1, uf=-2, degree=1)

v.assign(ic_expr)

File("foo/func.pvd") << v
