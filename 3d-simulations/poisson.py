import dolfin as df
from pathlib import Path


mesh = df.Mesh()
mesh_directory = Path.home() / "Documents" / "brain3d" / "meshes"
with df.XDMFFile(str(mesh_directory / "brain_64.xdmf")) as mesh_reader:
    mesh_reader.read(mesh)
mesh.coordinates()[:] /= 10     # Convert to cm

tensor_function_space = df.TensorFunctionSpace(mesh, "CG", 1)       # Use DG???
extracellular_function = df.Function(tensor_function_space)
directory = Path.home() / "Documents" / "brain3d" / "meshes"
name = "conductivity"  # TODO: change to conductivity
# with df.XDMFFile(str(directory / "foo.xdmf")) as ifh:
with df.XDMFFile(str(directory / "brain_64_intracellular_conductivity.xdmf")) as ifh:
    ifh.read_checkpoint(extracellular_function, name, counter=0)
# name = "indicator"  # TODO: change to conductivity
# with df.XDMFFile(str(directory / "extracellular_conductivity.xdmf")) as ifh:
#     ifh.read_checkpoint(extracellular_function, name, counter=0)

function_space = df.FunctionSpace(mesh, "CG", 1)
u = df.TrialFunction(function_space)
v = df.TestFunction(function_space)

dx = df.dx
F = df.inner(extracellular_function*df.grad(u), df.grad(v))*dx + df.Constant(1)*v*dx

a = df.lhs(F)
L = df.rhs(F)

A = df.assemble(a)
b = df.assemble(L)

bcs = df.DirichletBC(function_space, df.Constant(0), df.DomainBoundary())
bcs.apply(A, b)

solver = df.KrylovSolver("gmres", "petsc_amg")
solver.set_operator(A)

solution = df.Function(function_space)
solver.solve(solution.vector(), b)

print("||solution||_L2:", solution.vector().norm("l2"))

df.File("poisson.pvd") << solution
