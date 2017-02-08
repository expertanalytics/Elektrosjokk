# Modified from demo

from cbcpost import SolutionField, PostProcessor
from cbcbeat import *
from adex import AdExManual
import time
from collections import OrderedDict
from IPython import embed


def setup_general_parameters():
    """ Turn on FFC/FEniCS optimizations
    """
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    parameters["form_compiler"]["quadrature_degree"] = 3


def setup_application_parameters():
    """ Define paremeters for the problem and solvers
    """

    # Setup application parameters and parse from command-line
    application_parameters = Parameters("Application")
    application_parameters.add("T", 1e-1)        # End time  (ms)
    application_parameters.add("timestep", 1e-2) # Time step (ms)
    application_parameters.add("directory",
                               "results_%s" % time.strftime("%Y_%d%b_%Hh_%Mm"))
    application_parameters.parse()

    # Turn off adjoint functionality
    parameters["adjoint"]["stop_annotating"] = True
    info(application_parameters, True)
    return application_parameters


def setup_conductivities():
    return (2, 1)   # M_i, M_e


def setup_cell_model(application_parameters):
    """ Setup cell model with parameters
    """
    params = OrderedDict([("C", 281),           # Membrane capacitance (pF
                          ("g_L", 30),          # Leak conductance (ns)
                          ("E_L", -70.6),       # Leak reversal potential (mV)
                          ("V_T", -50.4),       # Spike threshold (mV)
                          ("Delta_T", 2),       # Slope factor (mV)
                          ("tau_w", 144),       # Adaptation time constant (ms)
                          ("a", 4),             # Subthreshold adaptation (nS)
                          ("b", 0.085)])        # Spike-triggered adaptation (nA)

    return AdExManual(params=params)


def setup_cardiac_model(application_parameters):
    # Initialize the computational domain in time and space
    time = Constant(0.0)
    #mesh = UnitCubeMesh(100, 100, 100)
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "../../convex_hull/erika_res32.h5", "r")
    hdf.read(mesh, "/mesh", False)
    hdf.close()

    # Setup conductivities
    (M_i, M_e) = setup_conductivities()

    # Setup cell model
    cell_model = setup_cell_model(application_parameters)

    # Define some simulation protocol (use cpp expression for speed)
    pulse = Expression("10*t*x[0]", t=time, degree=1)

    # Initialize cardiac model with the above input
    heart = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus=pulse)
    return heart


def main():
    application_parameters = setup_application_parameters()
    setup_general_parameters()

    brain = setup_cardiac_model(application_parameters)

    # Extract end time and time-step from application parameters
    T = application_parameters["T"]
    k_n = application_parameters["timestep"]

    # Since we know the time-step we want to use here, set it for the
    # sake of efficiency in the bidomain solver
    params = SplittingSolver.default_parameters()
    params["theta"] = 0.5
    params["CardiacODESolver"]["scheme"] = "GRL1"   # TODO: what is this

    params["pde_solver"] = "bidomain"
    #params["BidomainSolver"]["linear_solver_type"] = "direct"
    params["BidomainSolver"]["linear_solver_type"] = "iterative"
    params["BidomainSolver"]["algorithm"] = "cg"
    params["BidomainSolver"]["preconditioner"] = "petsc_amg"
    solver = SplittingSolver(brain, params=params)

    # Extract solution fields from solver
    (vs_, vs, vu) = solver.solution_fields()

    # Extract and assign initial condition
    vs_.assign(brain.cell_models().initial_conditions())

    # Set-up solve
    solutions = solver.solve((0, T), k_n)

    # Set-up PostProcessor

    pp = PostProcessor(dict(casedir="Results", clean_casedir=True))
    pp.store_mesh(brain.domain())
    pp.store_params(dict(application_parameters))

    solution_field_params = dict(save=True,
                                 save_as=["hdf5", "xdmf"],
                                 plot=False,
                                )

    pp.add_field(SolutionField("v", solution_field_params))
    pp.add_field(SolutionField("u", solution_field_params))

    theta = params["theta"]
    for i, (timestep, fields) in enumerate(solutions):
        (t0, t1) = timestep
        (vs_, vs, vur) = fields
        v, u = vu.split(deepcopy=True)

        pp.update_all({"v": lambda: v, "u" : lambda: u}, t0 + theta*(t1 - t0), i)

    pp.finalize_all()
    return solver


if __name__ == "__main__":
    main()
    print "Success!"
