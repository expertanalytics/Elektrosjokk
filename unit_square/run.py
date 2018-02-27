import xalbrain as xb 
import numpy as np

from cbcpost import (
    PostProcessor,
    Field,
    SolutionField
)


def setup_general_parameters():
    """Turn on FFC/FEniCS optimizations
    The options are borrowed from the cbcbeat demo. 
    """
    xb.parameters["form_compiler"]["representation"] = "uflacs"
    xb.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    xb.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    xb.parameters["form_compiler"]["quadrature_degree"] = 3


def setup_conductivities(mesh):
    Q = xb.FunctionSpace(mesh, "DG", 0)
    Mi = xb.Function(Q)
    Me = xb.Function(Q)

    Mi.vector()[:] = np.random.random(Mi.vector().array().size)
    Me.vector()[:] = np.random.random(Me.vector().array().size) + 3
    return Mi, Me


def setup_cell_model(application_parameters):
    """Define the cell model.
    Optionally change the default parameters of the cell model.
    """
    params = xb.Wei.default_parameters()
    return xb.Wei(params=params)


def setup_application_parameters():
    """Define parameters for the problem and solvers."""
    application_parameters = xb.Parameters("Application")
    application_parameters.add("T", 0.01)                # End time  (ms)
    application_parameters.add("timestep", 5e-3)        # Time step (ms)
    application_parameters.add("directory", "results")
    application_parameters.parse()
    return application_parameters


def setup_brain_model(application_parameters, N=10):
    """Return the xb model."""
    # Initialize the computational domain in time and space
    time = xb.Constant(0.0)       # All time dependencies must rely on this instance
    mesh = xb.UnitSquareMesh(N, N)

    # Setup conductivities
    (M_i, M_e) = setup_conductivities(mesh)     # The keys match cell_domains.array()

    # Setup cell model
    cell_model = setup_cell_model(application_parameters)

    # Define some external stimulus
    stimulus = xb.Expression("(x[0] > 0.9 && t <= 7.0) ? 180.0 : 0.0",
                               t=time, degree=3)    # time is same instance as in xb

    # Initialize brain model with the above input
    args = (mesh, time, M_i, M_e, cell_model)
    kwargs = {"stimulus": stimulus,
              "cell_domains": None,
              "facet_domains": None}
    brain = xb.CardiacModel(*args, **kwargs)
    return brain


def main():
    comm = xb.mpi_comm_world()
    application_parameters = setup_application_parameters()
    brain = setup_brain_model(application_parameters, N=10)

    # Customize and create a splitting solver
    splittingSolver_params = xb.SplittingSolver.default_parameters()

    splittingSolver_params["pde_solver"] = "bidomain"
    splittingSolver_params["BidomainSolver"]["linear_solver_type"] = "direct"
    splittingSolver_params["theta"] = 0.5    # Second order splitting scheme
    splittingSolver_params["CardiacODESolver"]["scheme"] = "ERK4"   # Choose wisely

    # splittingSolver_params["BidomainSolver"]["linear_solver_type"] = "iterative"
    # splittingSolver_params["BidomainSolver"]["algorithm"] = "cg"
    # splittingSolver_params["BidomainSolver"]["preconditioner"] = "petsc_amg"

    solver = xb.SplittingSolver(brain, params=splittingSolver_params)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(brain.cell_models().initial_conditions())

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
        print(f"({timestep})")
        (vs_, vs, vur) = fields

        # theta dependency due to the splitting scheme
        current_t = t0 + theta*(t1 - t0)    

        solutions = vur.split(deepcopy=True)
        v = solutions[0]
        u = solutions[1]
        postprocessor.update_all({"v": lambda: v, "u": lambda: u}, current_t, i)

        if xb.MPI.rank(comm) == 0:
            print("Solving time {0} out of {1}".format(current_t, T))
    postprocessor.finalize_all()


if __name__ == "__main__":
    setup_general_parameters()

    main()
    print("Success!")
