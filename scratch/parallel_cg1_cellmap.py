import dolfin as df
import numpy as np


# mesh = df.UnitSquareMesh(10, 10)
mesh = df.UnitCubeMesh(10, 10, 10)

cell_function = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
cell_function.set_all(1)
df.CompiledSubDomain("x[0] >= 0.5").mark(cell_function, 2)
print(np.unique(cell_function.array()))

# File("foo.pvd") << cell_function

dX = df.Measure("dx", domain=mesh, subdomain_data=cell_function)

CV = df.FunctionSpace(mesh, "CG", 1)
DV = df.FunctionSpace(mesh, "CG", 1)

u = df.TrialFunction(DV)
v = df.TestFunction(DV)

sol = df.Function(DV)
sol.vector().zero()     # Make sure it is initialised to zero


F = -u*v*dX(1) + df.Constant(1)*v*dX(1)
F += -u*v*dX(2) + df.Constant(2)*v*dX(2)
a = df.lhs(F)
L = df.rhs(F)

A = df.assemble(a, keep_diagonal=True)
A.ident_zeros()
b = df.assemble(L)

solver = df.KrylovSolver("cg", "petsc_amg")
solver.set_operator(A)
solver.solve(sol.vector(), b)


v_new = df.Function(CV)
v_new.interpolate(sol)
# v_new = df.project(sol, CV)

import numpy as np

v_new.vector()[:] = np.rint(v_new.vector().get_local())
print(np.unique(v_new.vector().get_local()))

df.File("foo.pvd") << v_new


"""
dX = df.Measure("dx", domain=mesh, subdomain_data=cell_function)

V = df.FunctionSpace(mesh, "DG", 0)
u = df.TrialFunction(V)
v = df.TestFunction(V)
sol = df.Function(V)
sol.vector().zero()     # Make sure it is initialised to zero

for subdomain_id, value in ic_dict.items():
    F = -u*v*dX(subdomain_id) + df.Constant(value)*v*dX(subdomain_id)
    a = df.lhs(F)
    L = df.rhs(F)

    A = df.assemble(a, keep_diagonal=True)
    A.ident_zeros()
    b = df.assemble(L)
    solver = df.KrylovSolver("cg", "petsc_amg")
    solver.set_operator(A)
    solver.solve(sol.vector(), b)

    VCG = df.FunctionSpace(mesh, "CG", 1)
    v_new = df.Function(VCG)
    v_new.interpolate(sol)

    Vp = vs_prev.function_space().sub(subfunction_index)
    merger = df.FunctionAssigner(Vp, VCG)
    merger.assign(vs_prev.sub(subfunction_index), v_new)
"""
