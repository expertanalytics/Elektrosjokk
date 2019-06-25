from dolfin import *

from conductivites import get_conductivities
from shock import get_shock

# mesh = Mesh("erika_res32.xml")
mesh = Mesh("merge.xml.gz")
F_ele = FiniteElement("CG", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, F_ele)
W = FunctionSpace(mesh, MixedElement((F_ele, F_ele)))

u, v = TrialFunctions(W)
w, q = TestFunctions(W)

# need to implement conductivities and shock 
Me, Mi = get_conductivities(3)
shock = get_shock(0)
# shock = Expression("1", degree=1)

shock_function = project(shock, V)
zero_function = project(Constant(0), V)

UV = Function(W)   # soluiton on current timestep
UV_ = Function(W)  # solution on previous timestep

dt = 0.01
T = 1.0
t = 0.0

alpha = Constant(0)     # 10
dtc = Constant(dt)

# a = (v - UE_)/dtc*w*dx + inner(Mi*grad(v), grad(w))*dx
# a += inner(Mi*grad(v), grad(q))*dx + inner((Mi + Me)*grad(u), grad(q))*dx

# a = u*w*dx + dtc*inner(Me*grad(u), grad(w))*dx
a = dtc*inner(Me*grad(u), grad(w))*dx
a += dtc*inner(Me*grad(v), grad(w))*dx
a += dtc*inner(Me*grad(u), grad(q))*dx
a += v*q*dx + dtc*inner(Mi*grad(v), grad(q))*dx

"""
p = u*w*dx
p += dtc*inner(Me*grad(u), grad(w))*dx
p += v*q*dx
p += dtc*inner(Mi*grad(v), grad(q))*dx
"""

A = assemble(a)
# P = assemble(p)

ufile = File("results/u.pvd")
vfile = File("results/v.pvd")

# assigner = FunctionAssigner(W, [V, V])
# assigner.assign(UV_, [shock_function, zero_function])

solver = PETScKrylovSolver("gmres", "petsc_amg")
# solver.set_operators(A, P)
solver.set_operator(A)


while t <= T:
    t += dt
    UE_, V_ = split(UV_)
    shock.t = t
    L = UE_*w*dx + dtc*shock*w*ds + dt*alpha*UE_*w*dx
    b = assemble(L)
    b -= b.sum()/b.size()
    print("foo", b.norm("l2"))
    solver.solve(UV.vector(), b)

    print("time ", t, UV.vector().norm("l2"))

    UV_.assign(UV) # update solution on previous time step with current solution 

    UE, V = UV.split()

    ufile << UE
    vfile << V
