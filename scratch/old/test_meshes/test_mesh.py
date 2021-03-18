from dolfin import *


class UpperBox(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] > 0.55) and (x[1] < 0.9) and (x[0] > 0.1) and (x[0] < 0.9)


class LowerBox(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] > 0.1) and (x[1] < 0.45) and (x[0] > 0.1) and (x[0] < 0.9)

mesh = UnitSquareMesh(20, 20)
mf = MeshFunction("size_t", mesh, 2)
mf.set_all(0)
LowerBox().mark(mf, 1)
UpperBox().mark(mf, 2)

upper_expr_str = ",".join((
    "(x[1] - 0.55)*(1*(x[0] > 0.5) - 1*(x[0] < 0.5))",
    "1 - 4*(x[0] - 0.5)*(x[0] - 0.5)"
))

lower_expr_str = ",".join((
    "(0.45 - x[1])*(1*(x[0] > 0.5) - 1*(x[0] < 0.5))",
    "4*(x[0] - 0.5)*(x[0] - 0.5) - 1"
))


upper_expression = Expression(upper_expr_str, degree=2)
lower_expression = Expression(lower_expr_str, degree=2)

F = FunctionSpace(mesh, "CG", 2)
upper_expr = interpolate(upper_expression, F)
lower_expr = interpolate(lower_expression, F)

File("foo.pvd") << upper_expr
File("bar.pvd") << lower_expr
