import pickle 
import xalbrain as xb 
import numpy as np

from wei_utils import (
    get_random_time,
    save_points,
    pickle_points,
    get_uniform_ic,
)

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

    # Mi.vector()[:] = np.random.random(Mi.vector().array().size)
    # Me.vector()[:] = 3*np.random.random(Me.vector().array().size)
    Mi.vector()[:] = 1
    Me.vector()[:] = 1
    return Mi, Me


def setup_cell_model(application_parameters):
    """Define the cell model.
    Optionally change the default parameters of the cell model.
    """
    params = xb.cellmodels.Wei.default_parameters()
    return xb.cellmodels.Wei(params=params)


def setup_application_parameters():
    """Define parameters for the problem and solvers."""
    application_parameters = xb.Parameters("Application")
    application_parameters.add("T", 10000.0)               # End time  (ms)
    application_parameters.add("timestep", 5e-2)        # Time step (ms)
    application_parameters.add("directory", "results")
    application_parameters.parse()
    return application_parameters


def setup_brain_model(application_parameters, N=10):
    """Return the xb model."""
    # Initialize the computational domain in time and space
    time = xb.Constant(0.0)       # All time dependencies must rely on this instance
    mesh = xb.UnitSquareMesh(N, N)
    # mesh.coordinates()[:] *= 10

    # Setup conductivities
    (M_i, M_e) = setup_conductivities(mesh)     # The keys match cell_domains.array()

    # Setup cell model
    cell_model = setup_cell_model(application_parameters)

    # Define some external stimulus
    stimulus = xb.Expression(
        "(2 < t && t < 5) ? 800*exp(-0.003*pow(x[0] - 0.5, 2))*exp(-0.003*pow(x[1] - 0.5, 2)) : 0.0",
        t=time,
        degree=1
    )

    # stimulus = xb.Expression(
    #     "(x[0] > 0.9 && t <= 7.0) ? 180.0 : 0.0",
    #     t=time, degree=3
    # )    # time is same instance as in xb

    # Initialize brain model with the above input
    args = (mesh, time, M_i, M_e, cell_model)
    kwargs = {
        "stimulus": None,
        "cell_domains": None,
        "facet_domains": None
    }
    brain = xb.CardiacModel(*args, **kwargs)
    return brain


def assign_ic(func):
    mixed_func_space = func.function_space()

    functions = func.split(deepcopy=True)
    ic = get_random_time(functions[0].vector().size())
    V = xb.FunctionSpace(mixed_func_space.mesh(), "CG", 1)

    for i, f in enumerate(functions):
        ic_func = xb.Function(V)
        ic_func.vector()[:] = np.array(ic[:, i])
    
        assigner = xb.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(func.sub(i), ic_func)


def main():
    application_parameters = setup_application_parameters()
    brain = setup_brain_model(application_parameters, N=10)

    # Customize and create a splitting solver
    splittingSolver_params = xb.SplittingSolver.default_parameters()

    # splittingSolver_params["pde_solver"] = "monodomain"
    splittingSolver_params["pde_solver"] = "bidomain"
    splittingSolver_params["theta"] = 0.5    # Second order splitting scheme
    splittingSolver_params["CardiacODESolver"]["scheme"] = "RK4"   # Choose wisely

    # splittingSolver_params["MonodomainSolver"]["linear_solver_type"] = "iterative"
    # splittingSolver_params["MonodomainSolver"]["algorithm"] = "cg"
    # splittingSolver_params["MonodomainSolver"]["preconditioner"] = "petsc_amg"

    splittingSolver_params["BidomainSolver"]["linear_solver_type"] = "iterative"
    splittingSolver_params["BidomainSolver"]["algorithm"] = "gmres"
    splittingSolver_params["BidomainSolver"]["preconditioner"] = "petsc_amg"
    splittingSolver_params["BidomainSolver"]["use_avg_u_constraint"] = False 
    splittingSolver_params["apply_stimulus_current_to_pde"] = False    # Second order splitting scheme

    splittingSolver_params["BidomainSolver"]["petsc_krylov_solver"]["absolute_tolerance"] = 1e-14
    splittingSolver_params["BidomainSolver"]["petsc_krylov_solver"]["relative_tolerance"] = 1e-14
    splittingSolver_params["BidomainSolver"]["petsc_krylov_solver"]["nonzero_initial_guess"] = True
    # for key, value in splittingSolver_params["BidomainSolver"]["petsc_krylov_solver"].items():
    #     print(key, value)
    # assert False

    solver = xb.SplittingSolver(brain, params=splittingSolver_params)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    # vs_.assign(brain.cell_models().initial_conditions())

    brain.cell_models().set_initial_conditions(**get_uniform_ic("spike"))
    vs_.assign(brain.cell_models().initial_conditions())
    # assign_ic(vs_)

    values = {str(f): f(0.5, 0.5) for f in vs_.split()}
    with open("INITIAL_CONDITION.pickle", "wb") as out_handle:
        pickle.dump(values, out_handle)

    # Extract end time and time-step from application parameters
    T = application_parameters["T"]
    k_n = application_parameters["timestep"]

    postprocessor = PostProcessor(dict(casedir="test", clean_casedir=True))
    postprocessor.store_mesh(brain.domain())

    field_params = dict(
        save=True,
        save_as=["hdf5", "xdmf"],
        plot=False,
        start_timestep=-1,
        stride_timestep=1
    )

    postprocessor.add_field(SolutionField("v", field_params))
    # postprocessor.add_field(SolutionField("u", field_params))

    # myfile = xb.File("last2.pvd")

    theta = splittingSolver_params["theta"]

    # Solve forward problem
    for i, (timestep, fields) in enumerate(solver.solve((0, T), k_n)):
        t0, t1 = timestep
        # print(f"({timestep})")
        (vs_, vs, vur) = fields

        functions = vs.split()
        # values = [f(0.5, 0.5) for f in functions]
        # print_str = "{:.3e} "*len(values)
        # print(print_str.format(*values))

        # theta dependency due to the splitting scheme
        current_t = t0 + theta*(t1 - t0)

        # v, u = vur.split(deepcopy=True)
        # v = vur.split(deepcopy=True)

        # myfile << functions[4]

        if i % 100 == 0:
            # postprocessor.update_all({"v": lambda: v, "u": lambda: u}, current_t, i)
            postprocessor.update_all({"v": lambda: vur}, current_t, i)
            print(i, vur.vector().norm("l2"), flush=True)
            yield current_t, functions

    postprocessor.finalize_all()


if __name__ == "__main__":
    setup_general_parameters()

    pickle_points(main(), ((0.5, 0.5),), "time_samples")
    print("Success!")
