import dolfin as df


mesh = df.Mesh()
with df.XDMFFile("mesh/brain_128.xdmf") as mesh_reader:
    mesh_reader.read(mesh)


function_space = df.FunctionSpace(mesh, "CG", 1)

u = df.TrialFunction(function_space)
v = df.TestFunction(function_space)

dx = df.dx
F = u*v*dx + df.Constant(1)*v*dx

a = df.lhs(F)
L = df.rhs(F)

A = df.assemble(a)
b = df.assemble(L)

solver = df.KrylovSolver("cg", "petsc_amg")
solver.set_operator(A)

solution = df.Function(function_space)
solver.solve(solution.vector(), b)

print("||solution||_L2:", solution.vector().norm("l2"))
