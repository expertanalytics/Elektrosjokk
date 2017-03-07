#!/usr/bin/env python
#  -*- coding: utf-8 -*-


# Import the cbcbeat module
from cbcpost import PostProcessor, Field, SolutionField
import cbcbeat as beat
import numpy.random
comm = beat.mpi_comm_world()

def setup_general_parameters():
    """Turn on FFC/FEniCS optimizations
    The options are borrowed from the cbcbeat demo. 
    """
    beat.parameters["form_compiler"]["representation"] = "uflacs"
    beat.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    beat.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    beat.parameters["form_compiler"]["quadrature_degree"] = 3


def setup_application_parameters():
    """Define parameters for the problem and solvers.
    """
    application_parameters = beat.Parameters("Application")
    application_parameters.add("T", 1.5)                # End time  (ms)
    application_parameters.add("timestep", 5e-3)        # Time step (ms)
    application_parameters.add("directory", "results")
    application_parameters.parse()

    # Turn off adjoint functionality
    beat.parameters["adjoint"]["stop_annotating"] = True
    beat.info(application_parameters, True)
    return application_parameters


def setup_conductivities(mesh):
    # Create some random cunductivity over two subdomains
    Q = beat.FunctionSpace(mesh, "DG", 0)
    Mi_low = beat.Function(Q)
    Mi_low.vector()[:] = 0.1*(numpy.random.rand(Q.dim()) + 1.0)

    Me_low = beat.Function(Q)
    Me_low.vector()[:] = 0.1*(numpy.random.rand(Q.dim()) + 1.0)

    Mi_high = beat.Function(Q)
    Mi_high.vector()[:] = 0.1*(numpy.random.rand(Q.dim()) + 5.0)

    Me_high = beat.Function(Q)
    Me_high.vector()[:] = 0.1*(numpy.random.rand(Q.dim()) + 5.0)

    # NB! Keys has to match the tags in cardiac_model.cell_domains() 
    M_i = {0: Mi_low, 1: Mi_high}
    M_e = {0: Me_low, 1: Me_high}

    return M_i, M_e


def setup_cell_model(application_parameters):
    """Define the cell model.
    Optionally change the default parameters of the cell model.
    """
    params = beat.AdExManual.default_parameters()
    return beat.AdExManual(params=params)


def setup_brain_model(application_parameters, N=10):
    """Return the brain model.
    """
    # Initialize the computational domain in time and space
    time = beat.Constant(0.0)       # All time dependencies must rely on this instance
    mesh = beat.UnitCubeMesh(N, N, N)

    # mesh function to keep track of the conductivities
    cell_domains = beat.CellFunction("size_t", mesh)
    cell_domains.set_all(0)

    # mark one corner
    corner = beat.CompiledSubDomain("x[0] > 0.5 && x[1] > 0.5 && x[2] > 0.5")
    corner.mark(cell_domains, 1)

    # Setup conductivities
    (M_i, M_e) = setup_conductivities(mesh)     # The keys match cell_domains.array()

    # Setup cell model
    cell_model = setup_cell_model(application_parameters)

    # Define some external stimulus
    stimulus = beat.Expression("(x[0] > 0.9 && t <= 7.0) ? 180.0 : 0.0",
                               t=time, degree=3)    # time is same instance as in brain

    # Initialize brain model with the above input
    args = (mesh, time, M_i, M_e, cell_model)
    kwargs = {"stimulus": stimulus,
              "cell_domains": cell_domains,
              "facet_domains": None}
    brain = beat.CardiacModel(*args, **kwargs)
    return brain


def main():
    comm = beat.mpi_comm_world()
    setup_general_parameters()
    application_parameters = setup_application_parameters()
    brain = setup_brain_model(application_parameters)

    # Customize and create a splitting solver
    splittingSolver_params = beat.SplittingSolver.default_parameters()
    
    splittingSolver_params["theta"] = 0.5    # Second order splitting scheme
    splittingSolver_params["pde_solver"] = "bidomain"
    splittingSolver_params["CardiacODESolver"]["scheme"] = "ERK4"   # Choose wisely
    splittingSolver_params["BidomainSolver"]["linear_solver_type"] = "iterative"
    splittingSolver_params["BidomainSolver"]["algorithm"] = "cg"
    splittingSolver_params["BidomainSolver"]["preconditioner"] = "petsc_amg"

    solver = beat.SplittingSolver(brain, params=splittingSolver_params)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(brain.cell_models().initial_conditions(), solver.VS)

    # Extract end time and time-step from application parameters
    T = application_parameters["T"]
    k_n = application_parameters["timestep"]

    postprocessor = PostProcessor(dict(casedir="test", clean_casedir=True))
    postprocessor.store_mesh(brain.domain())

    field_params = dict(save=True,
                        save_as=["hdf5", "xdmf"],
                        plot=False,
                        start_timestep=-1,
                        stride_timestep=1
                       )

    postprocessor.add_field(SolutionField("v", field_params))
    postprocessor.add_field(SolutionField("u", field_params))
    theta = splittingSolver_params["theta"]

    # Solve forward problem
    for i, (timestep, fields) in enumerate(solver.solve((0, T), k_n)):
        t0, t1 = timestep
        print "(t_0, t_1) = (%g, %g)" % timestep
        (vs_, vs, vur) = fields

        # theta dependency due to the splitting scheme
        current_t = t0 + theta*(t1 - t0)    

        solutions = vur.split(deepcopy=True)
        v = solutions[0]
        u = solutions[1]

        postprocessor.update_all({"v": lambda: v, "u": lambda: u}, current_t, i)
                                 
        if beat.MPI.rank(comm) == 0:
            print "Solving time {0} out of {1}".format(current_t, T)

    postprocessor.finalize_all()


if __name__ == "__main__":
    main()
