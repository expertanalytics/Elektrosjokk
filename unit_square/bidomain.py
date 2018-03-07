from xalbrain import BidomainSolver
from xalbrain.dolfinimport import *


from dolfin import set_log_level
set_log_level(1)


def main(N):
    mesh = UnitSquareMesh(N, N)
    time = Constant(0)

    Mi = Constant(1) 
    Me = Constant(1)

    bidomain_parameters = BidomainSolver.default_parameters()
    bidomain_parameters["linear_solver_type"] = "iterative"
    bidomain_parameters["algorithm"] = "gmres"
    bidomain_parameters["preconditioner"] = "petsc_amg"
    bidomain_parameters["use_avg_u_constraint"] = False

    ac_str = "cos(t)*cos(2*pi*x[0])*cos(2*pi*x[1]) + 4*pow(pi, 2)*cos(2*pi*x[0])*cos(2*pi*x[1])*sin(t)"
    stimulus = Expression(ac_str, t=time, degree=3)
    
    solver = BidomainSolver(mesh, time, Mi, Me, I_s=stimulus, params=bidomain_parameters)

    for t, (v, vur) in solver.solve((0, 1), dt=0.1):
        print(v.vector().norm("l2"), vur.vector().norm("l2"))



if __name__ == "__main__":
    main(10)
