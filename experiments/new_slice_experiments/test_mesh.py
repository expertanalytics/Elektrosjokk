import dolfin as df
from coupled_utils import get_mesh


#mesh, cell_function, interface_function = get_mesh("new_meshes", "skullgmwm")


mesh = df.Mesh()
with df.XDMFFile("new_meshes/skullgmwm.xdmf") as infile:
    infile.read(mesh)

# mesh = df.Mesh("new_meshes/skullgmwm.xml")
# from IPython import embed; embed()

# print(set(cell_function.array()))
# print(set(interface_function.array()))
# print(mesh.geometry().dim())


V = df.FunctionSpace(mesh, "CG", 1)
u = df.TrialFunction(V)
v = df.TestFunction(V)

F = df.inner(df.grad(u), df.grad(v))*df.dx + df.Constant(1)*v*df.dx

a = df.lhs(F)
L = df.rhs(F)

# from IPython import embed; embed()

solution = df.Function(V)
bc = df.DirichletBC(V, df.Constant(0), df.DomainBoundary())

df.solve(a == L, solution, bc)
print(solution.vector().norm("l2"))
