# Modified from demo

from cbcpost import SolutionField, PostProcessor
from cbcbrain import *
from cbcbrain.cellmodels import AdExManual
import time
from collections import OrderedDict
from IPython import embed
from shock import get_shock
from conductivites import *


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
    application_parameters.add("timestep", 1e-2)        # Time step (ms)
    application_parameters.add("directory",
                               "results_%s" % time.strftime("%Y_%d%b_%Hh_%Mm"))
    application_parameters.parse()

    # Turn off adjoint functionality
    parameters["adjoint"]["stop_annotating"] = True
    info(application_parameters, True)
    return application_parameters


def setup_conductivities():
    """
    Returns
    -------
    (M_i, M_e)
    """

    intracellular = IntracellularConductivity(degree=1)
    extracellular = ExtracellularConductivity(degree=1)
    water = Water(degree=1)

    M_i = {1: intracellular, 2: intracellular}
    M_e = {1: extracellular, 2: extracellular}
    #M_i = intracellular
    #M_e = extracellular

    return M_i, M_e


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
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "../../convex_hull/brain.h5", "r")
    hdf.read(mesh, "/mesh", False)

    facet_domains = FacetFunction("size_t", mesh)
    hdf.read(facet_domains, "/boundaries")

    cell_domains = CellFunction("size_t", mesh)
    hdf.read(cell_domains, "/domains")
    print "Cell domainds.array() :", set(cell_domains.array())

    # Setup conductivities
    (M_i, M_e) = setup_conductivities()

    # Setup cell model
    cell_model = setup_cell_model(application_parameters)

    # Define some simulation protocol (use cpp expression for speed)
    #pulse = Expression("10*t*x[0]", t=time, degree=1)
    pulse = get_shock()

    # Initialize cardiac model with the above input
    args = (mesh, time, M_i, M_e, cell_model)
    kwargs = {"stimulus" : pulse, "cell_domains" : cell_domains, "facet_domains" : facet_domains}
    #kwargs = {"stimulus" : pulse}
    heart = CardiacModel(*args, **kwargs)
    return heart


def main():
    application_parameters = setup_application_parameters()
    setup_general_parameters()

    brain = setup_cardiac_model(application_parameters)

    # Extract end time and time-step from application parameters
    T = application_parameters["T"]
    k_n = application_parameters["timestep"]

    splitting_solver_params = SplittingSolver.default_parameters()
    splitting_solver_params["theta"] = 0.5
    splitting_solver_params["CardiacODESolver"]["scheme"] = "GRL1"   # TODO: what is this

    splitting_solver_params["pde_solver"] = "bidomain"
    #splitting_solver_params["BidomainSolver"]["linear_solver_type"] = "direct"
    splitting_solver_params["BidomainSolver"]["linear_solver_type"] = "iterative"
    splitting_solver_params["BidomainSolver"]["algorithm"] = "cg"
    splitting_solver_params["BidomainSolver"]["preconditioner"] = "petsc_amg"
    solver = SplittingSolver(brain, params=splitting_solver_params)

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

    theta = splitting_solver_params["theta"]
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
