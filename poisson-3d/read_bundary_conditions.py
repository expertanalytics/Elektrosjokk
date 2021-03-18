import dolfin as df

from post import Loader
from postspec import LoaderSpec

from pathlib import Path

casedir = Path("e9591a1d")
loader_spec = LoaderSpec(casedir=casedir)
loader = Loader(loader_spec)
# mesh = loader.load_mesh()
# boundarymesh = df.BoundaryMesh(mesh, "exterior")

# hull_mesh = df.Mesh()
# with df.XDMFFile(df.MPI.comm_world, "hull_128.xdmf") as mesh_file:
#     mesh_file.read(hull_mesh)
hull_mesh = df.Mesh("brain_128.xml")

print(hull_mesh.coordinates())


class Boundary_condition(df.UserExpression):
    def set_function(self, func):
        self.bc_func = func

    def eval(self, values, x):
        try:
            values[:] = self.bc_func(x)
        except:
            values[:] = 0


V_hull = df.FunctionSpace(hull_mesh, "CG", 1)
u = df.TrialFunction(V_hull)
v = df.TestFunction(V_hull)

a = df.inner(df.grad(u), df.grad(v))*df.dx
L = df.Constant(1)*v*df.dx

A = df.assemble(a)
b = df.assemble(L)

solver = df.KrylovSolver("cg", "amg")
solver.set_operator(A)


bc_class = Boundary_condition()
for t, f in loader.load_checkpoint("v"):
    print(t, f.vector().norm("l2"))
    bc_class.set_function(f)

    # Do not use DomainBoundary -- something more specific 
    bc = df.DirichletBC(V_hull, bc_class, df.DomainBoundary())
    bc.apply(A, b)

    us = df.Function(V_hull)
    solver.solve(us.vector(), b)
    print(us.vector().norm("l2"))

    xdmf_file = df.XDMFFile(df.MPI.comm_world, f"test_bc.xdmf")
    xdmf_file.write(us, 0.0)

    break
