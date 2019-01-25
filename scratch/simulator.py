from dolfin import *
from conductivites import get_conductivities
from shock import get_shock 

mesh = Mesh("erika_res32.xml")
# mesh = UnitCubeMesh(20, 20, 20) 
#E1 = FiniteElement("Lagrange", mesh.ufl_cell(),  1) # make scalar finite element
#E12 = E1*E1                                         # make two elements, one for each unknown 
#W = FunctionSpace(mesh, E12)                        # make finite element space  

F_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, F_ele)
W = FunctionSpace(mesh, MixedElement((F_ele, F_ele)))

#V = FunctionSpace(mesh, "Lagrange", 1) 
#W = MixedFunctionSpace((V, V))


ue, v = TrialFunctions(W) 
k, l = TestFunctions(W) 

# need to implement conductivities and shock 
Me, Mie = get_conductivities(3)
shock = get_shock(0)

shock_function = project(shock, V)
#plot(shock_function) 
#interactive()
zero_function = project(Constant(0), V)

dt = 0.01
T = 1.0
t = 0.0 
alpha = Constant(1)     # 10

dtc = Constant(dt)

a = ue*k*dx  + dtc*inner(Me*grad(ue), grad(k))*dx  \
             + dtc*inner(Me*grad(v), grad(k))*dx  \
             + dtc*inner(Me*grad(ue), grad(l))*dx  \
    +v*l*dx  + dtc*inner(Mie*grad(v), grad(l))*dx  \

# p =   ue*k*dx  + dtc*inner(dot(Me,grad(ue)), grad(k))*dx  \
#     + v*l*dx   + dtc*inner(dot(Mie,grad(v)), grad(l))*dx  

p =   ue*k*dx  + dtc*inner(Me*grad(ue), grad(k))*dx  \
    + v*l*dx   + dtc*inner(Mie*grad(v), grad(l))*dx  

A = assemble(a) 
P = assemble(p) 


UEV = Function(W)   # soluiton on current timestep  
UEV_ = Function(W)  # solution on previous timestep  
uefile = File("ue.pvd") 
vfile = File("v.pvd") 

assigner = FunctionAssigner(W, [V, V])
assigner.assign(UEV_, [shock_function, zero_function])

solver = PETScKrylovSolver("gmres", "petsc_amg")
solver.set_operators(A, P)

while t <= T: 
  t += dt 
  UE_, V_ = split(UEV_)  
  shock.t = t 
  L = UE_*k*dx + dtc*shock*k*ds  + dt*alpha*UE_*k*dx  
  b = assemble(L) 
  b -= b.sum()/b.size()
  print("foo", b.norm("l2"))
  solver.solve(UEV.vector(), b) 

  print("time ", t, UEV.vector().norm("l2"))

  UEV_.assign(UEV) # update solution on previous time step with current solution 

  UE, V = UEV.split() 
#  plot(UE)
#  interactive()

  uefile << UE
  vfile << V 


  


