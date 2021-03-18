import dolfin as df
from pathlib import Path


mesh = df.Mesh()
mesh_name = "brain_64"
mesh_directory = Path.home() / "Documents" / "brain3d" / "meshes"
with df.XDMFFile(str(mesh_directory / f"{mesh_name}.xdmf")) as mesh_reader:
    mesh_reader.read(mesh)
mesh.coordinates()[:] /= 10     # Convert to cm

function_space = df.FunctionSpace(mesh, "CG", 1)
zero_function = df.Function(function_space)
zero_function.vector()[:] = 0

one_function = df.Function(function_space)
one_function.vector()[:] = 1

# foobar = df.as_matrix([
#     [one_function, zero_function, zero_function],
#     [zero_function, one_function, zero_function],
#     [zero_function, zero_function, one_function]
# ])

# foo = foobar.spit(deepcopy=True)
# for v in foo:
#     print(v.vector().get_local().sum())


# assert False, "Success"

# tensor_function_space = df.TensorFunctionSpace(mesh, "CG", 1)       # Use DG???
tensor_function_space = df.TensorFunctionSpace(mesh, "DG", 0)       # Use DG???

extracellular_function = df.Function(tensor_function_space)
directory = Path.home() / "Documents" / "brain3d" / "meshes"
name = "conductivity"  # TODO: change to conductivity
# with df.XDMFFile(str(directory / "foo.xdmf")) as ifh:
with df.XDMFFile(str(directory / f"{mesh_name}_intracellular_conductivity.xdmf")) as ifh:
    ifh.read_checkpoint(extracellular_function, name, counter=0)
# name = "indicator"  # TODO: change to conductivity
# with df.XDMFFile(str(directory / "extracellular_conductivity.xdmf")) as ifh:
#     ifh.read_checkpoint(extracellular_function, name, counter=0)


foo = extracellular_function.split(deepcopy=True)

foobar = df.as_matrix([
    [foo[0], foo[1], foo[2]],
    [foo[3], foo[4], foo[5]],
    [foo[6], foo[7], foo[8]]
])

for v in foo:
    print(v.vector().get_local().sum())

# profunc = df.Function(tensor_function_space)
# profunc = df.project(foobar, tensor_function_space)

# with df.XDMFFile("foobar.xdmf") as foofile:
#     foofile.write_checkpoint(profunc, "foobar", 0)

# from IPython import embed; embed()
assert False

u = df.TrialFunction(function_space)
v = df.TestFunction(function_space)

dx = df.dx
F = df.inner(foobar*df.grad(u), df.grad(v))*dx + df.Constant(1)*v*dx
# F = df.inner(extracellular_function*df.grad(u), df.grad(v))*dx + df.Constant(1)*v*dx
# F = df.inner(df.grad(u), df.grad(v))*dx + df.Constant(1)*v*dx

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

with df.XDMFFile("poisson.xdmf") as out_file:
    out_file.write_checkpoint(solution, "poisson", 0)
# df.File("poisson.pvd") << solution
