# Modified from demo

from cbcpost import PostProcessor, Field, SolutionField, Restrict
from cbcpost.utils import create_submesh
import xalbrain
from xalbrain.cellmodels import AdExManual
import time
from collections import OrderedDict
from shock import Shock3D
from conductivites import IntracellularConductivity, ExtracellularConductivity, Water

import dolfin


def setup_general_parameters():
    """Turn on FFC/FEniCS optimizations."""
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 3


def setup_application_parameters():
    """Define parameters for the problem and solvers."""
    application_parameters = dolfin.Parameters("Application")
    application_parameters.add("T", 3e0)                # End time  (ms)
    application_parameters.add("timestep", 1e-2)        # Time step (ms)
    application_parameters.add("directory",
                               "results_%s" % time.strftime("%Y_%d%b_%Hh_%Mm"))
    application_parameters.parse()

    # Turn off adjoint functionality
    beat.parameters["adjoint"]["stop_annotating"] = True
    beat.info(application_parameters, True)
    return application_parameters


def setup_conductivities():
    """Expressions for extra- and intracellular conductance. 

    The extra- and intracellulas conductances are returned as two dictionaries with
    keys corresponding to a mesh function.

    Returns:
        M_i : dict
            {tag, Expression}
        M_e : dict
            {tag, Expression}
    """

    intracellular = IntracellularConductivity(degree=1)
    extracellular = ExtracellularConductivity(degree=1)
    water = Water(degree=1)

    M_i = {1: intracellular, 2: intracellular*1e-6}
    M_e = {1: extracellular, 2: water}

    return M_i, M_e


def setup_cell_model():
    """Setup cell model with parameters.
    
    Returns:
        CardiacCellModel
    """
    params = OrderedDict([("C", 281e-3),        # Membrane capacitance (nF)
                          ("g_L", 30e-3),       # Leak conductance (\mu S)
                          #("g_L", 30),         # Leak conductance (nS)
                          ("E_L", -70.6),       # Leak reversal potential (mV)
                          ("V_T", -50.4),       # Spike threshold (mV)
                          ("Delta_T", 2),       # Slope factor (mV)
                          ("tau_w", 144),       # Adaptation time constant (ms)
                          ("a", 4e-3),          # Subthreshold adaptation (\mu S)
                          #("a", 4),            # Subthreshold adaptation (nS)
                          ("b", 80.5e-3)])      # Spike-triggered adaptation (mA)
                          #("b", 0.0805)])      # Spike-triggered adaptation (nA)

    return AdExManual(params=params)


def setup_brain_model(application_parameters):
    """Return brain model instance.
    
    The mesh, `cell_domains` and `facet_domains` are loaded from a hdf5 file.

    Returns:
        CardiacModel
    """

    # Initialize the computational domain in time and space
    time = beat.Constant(0.0)
    mesh = beat.Mesh()
    hdf = beat.HDF5File(mesh.mpi_comm(), "../convex_hull/brain.h5", "r")
    hdf.read(mesh, "/mesh", False)

    facet_domains = beat.FacetFunction("size_t", mesh)
    hdf.read(facet_domains, "/boundaries")

    cell_domains = beat.CellFunction("size_t", mesh)
    hdf.read(cell_domains, "/domains")

    # Setup conductivities
    (M_i, M_e) = setup_conductivities()

    # Setup cell model
    cell_model = setup_cell_model()

    # Define some simulation protocol (use cpp expression for speed)
    shock_kwargs = {"t": time,
                    "spread": 0.003,
                    "amplitude": 8e2,
                    "center": (31, -15, 73),
                    "half_period": 1e-1,
                    "degree": 1}

    pulse = Shock3D(**shock_kwargs)

    # Initialize brain model with the above input
    args = (mesh, time, M_i, M_e, cell_model)
    kwargs = {"stimulus": pulse, "cell_domains": cell_domains,
              "facet_domains": facet_domains}
    brain = beat.CardiacModel(*args, **kwargs)
    return brain


def main():
    comm = beat.mpi_comm_world()
    application_parameters = setup_application_parameters()
    setup_general_parameters()

    brain = setup_brain_model(application_parameters)

    # Extract end time and time-step from application parameters
    T = application_parameters["T"]
    k_n = application_parameters["timestep"]

    splitting_solver_params = beat.SplittingSolver.default_parameters()
    splitting_solver_params["theta"] = 0.5
    splitting_solver_params["CardiacODESolver"]["scheme"] = "ERK1"  # choose wisely

    splitting_solver_params["pde_solver"] = "bidomain"
    splitting_solver_params["BidomainSolver"]["linear_solver_type"] = "iterative"
    splitting_solver_params["BidomainSolver"]["algorithm"] = "cg"
    splitting_solver_params["BidomainSolver"]["preconditioner"] = "petsc_amg"
    solver = beat.SplittingSolver(brain, params=splitting_solver_params)

    # Extract solution fields from solver
    (vs_, vs, vur) = solver.solution_fields()

    # Extract and assign initial condition
    vs_.assign(brain.cell_models().initial_conditions())

    # Set-up solve
    solutions = solver.solve((0, T), k_n)

    # Set-up PostProcessor

    postprocessor = PostProcessor(dict(casedir=application_parameters["directory"],
                                       clean_casedir=True))
    postprocessor.store_mesh(brain.domain())
    postprocessor.store_params(dict(application_parameters))

    field_params = dict(save=True,
                        save_as=["hdf5", "xdmf"],
                        plot=False,
                        start_timestep=-1,
                        stride_timestep=1
                       )

    # Add transmembrane potential, v, and extracellular potential, u.
    postprocessor.add_field(SolutionField("v", field_params))
    postprocessor.add_field(SolutionField("u", field_params))

    # Add fields over submeshes, defined by tags in brain.cell_domains()
    brainrestrictor = create_submesh(brain.domain(), brain.cell_domains(), 1)
    waterrestrictor = create_submesh(brain.domain(), brain.cell_domains(), 2)

    postprocessor.add_fields([
          Restrict("u", brainrestrictor, field_params, name="u_brain"),
          Restrict("v", brainrestrictor, field_params, name="v_brain"),
          Restrict("u", waterrestrictor, field_params, name="u_water"),
                             ])

    theta = splitting_solver_params["theta"]
    for i, (timestep, fields) in enumerate(solutions):
        (t0, t1) = timestep
        (vs_, vs, vur) = fields
        solutions = vur.split(deepcopy=True)
        v = solutions[0]
        u = solutions[1]

        current_t = t0 + theta*(t1 - t0)
        postprocessor.update_all({"v": lambda: v,   # Update and store all fields
                                  "u": lambda: u},
                                 current_t, i)
        if beat.MPI.rank(comm) == 0:
            print "Solving time {0} out of {1}".format(current_t, T)

    postprocessor.finalize_all()


if __name__ == "__main__":
    beat.set_log_level(100)
    main()
    print "Success!"
