#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# A basic practical example of how to use the cbcbeat module, in
# particular how to solve the bidomain equations coupled to a
# moderately complex cell model using the splitting solver provided by
# cbcbeat and to compute a sensitivity. 
#
# Sensitivity example for cbcbeat 
# ===============================


# Import the cbcbeat module
from cbcbeat import *
import numpy.random
from cbcpost import *
comm = mpi_comm_world()

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Define the computational domain
N = 20
mesh = UnitCubeMesh(N, N, N)
time = Constant(0.0)

# Create synthetic conductivity
Q = FunctionSpace(mesh, "DG", 0)
M_i = Function(Q)
M_i.vector()[:] = 0.1*(numpy.random.rand(Q.dim()) + 1.0)

M_e = Function(Q)
M_e.vector()[:] = 0.1*(numpy.random.rand(Q.dim()) + 1.0)

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = AdExManual()

# Define some external stimulus
stimulus = Expression("(x[0] > 0.9 && t <= 7.0) ? 360.0 : 0.0",
                      t=time, degree=3)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5                        # Second order splitting scheme
ps["pde_solver"] = "bidomain"          # Use Monodomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "GRL1" # 1st order Rush-Larsen for the ODEs
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["algorithm"] = "cg"
ps["BidomainSolver"]["preconditioner"] = "petsc_amg"

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions(), solver.VS)

# Time stepping parameters
k = 0.005
T = 0.3
interval = (0.0, T)


postprocessor = PostProcessor(dict(casedir="test",
                                   clean_casedir=True))
postprocessor.store_mesh(cardiac_model.domain())

field_params = dict(save=True,
                    save_as=["hdf5", "xdmf"],
                    plot=False,
                    start_timestep=-1,
                    stride_timestep=1
                   )

postprocessor.add_field(SolutionField("v", field_params))
postprocessor.add_field(SolutionField("u", field_params))
theta = 0.5

# Solve forward problem
for i, (timestep, fields) in enumerate(solver.solve(interval, k)):
    t0, t1 = timestep
    print "(t_0, t_1) = (%g, %g)" % timestep
    (vs_, vs, vur) = fields

    current_t = t0 + theta*(t1 - t0)

    solutions = vur.split(deepcopy=True)
    v = solutions[0]
    u = solutions[1]

    postprocessor.update_all({"v": lambda: v, "u": lambda: u},
                             current_t, i)
    if MPI.rank(comm) == 0:
        print "Solving time {0} out of {1}".format(current_t, T)

postprocessor.finalize_all()
