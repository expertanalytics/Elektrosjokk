import mpi4py

import xalbrain as xb
import numpy as np

from ect.pdesolvers import (
    setup_conductivities,
)

from ect.specs import (
    PDESimulationSpec,
    SolutionFieldSpec,
)

from ect.utilities import (
    wei_uniform_ic,
    create_dataframe,
    NonuniformIC,
    assign_ic,
)

from cbcpost import (
    PostProcessor,
    Field,
    SolutionField,
)


np.random.seed(42)
COMM = xb.mpi_comm_world().tompi4py()


def setup_general_parameters():
    """Turn on FFC/FEniCS optimizations from the cbcbeat demo."""
    xb.parameters["form_compiler"]["representation"] = "uflacs"
    xb.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    xb.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    xb.parameters["form_compiler"]["quadrature_degree"] = 3


def make_brain():
    """Create a CardiacModel for storing relevant parameters."""
    time = xb.Constant(0)

    N = 200
    mesh = xb.UnitSquareMesh(N, N)
    M_i, M_e = setup_conductivities(mesh)
    cell_model = xb.cellmodels.Wei()
    args = (mesh, time, M_i, M_e, cell_model)
    return xb.CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus=None)


def main():
    brain = make_brain() 
    field_spec = SolutionFieldSpec()
    simulation_spec = PDESimulationSpec(end_time=1e-0, timestep=1e-3)

    sparams = xb.SplittingSolver.default_parameters()
    sparams["pde_solver"] = "bidomain"
    sparams["theta"] = 0.5
    sparams["CardiacODESolver"]["scheme"] = "RK4"
    sparams["BidomainSolver"]["linear_solver_type"] = "iterative"
    sparams["BidomainSolver"]["algorithm"] = "gmres"
    sparams["BidomainSolver"]["preconditioner"] = "petsc_amg"
    sparams["BidomainSolver"]["use_avg_u_constraint"] = False
    sparams["apply_stimulus_current_to_pde"] = False
    sparams["BidomainSolver"]["petsc_krylov_solver"]["absolute_tolerance"] = 1e-11
    sparams["BidomainSolver"]["petsc_krylov_solver"]["relative_tolerance"] = 1e-11
    sparams["BidomainSolver"]["petsc_krylov_solver"]["nonzero_initial_guess"] = True
    solver = xb.SplittingSolver(brain, params=sparams)

    # Set initial conditions
    vs_, vs, vur = solver.solution_fields()
    data = np.load("REFERENCE_SOLUTION.npy")
    assign_ic(vs_, data)

    postprocessor = PostProcessor(dict(casedir="_test", clean_casedir=True))
    postprocessor.store_mesh(brain.mesh)
    postprocessor.add_field(SolutionField("v", field_spec._asdict()))
    postprocessor.add_field(SolutionField("u", field_spec._asdict()))
    postprocessor.add_field(SolutionField("vs", field_spec._asdict()))
    theta = sparams["theta"]

    end_time = simulation_spec.end_time
    dt = simulation_spec.timestep
    for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, end_time), dt)):
        print(f"Solving ({t0}, {t1})")
        functions = vs.split()
        current_t = t0 + theta*(t1 - t0)
        v, u = vur.split(deepcopy=True)

        postprocessor.update_all(
            {"v": lambda: v, "u": lambda: u, "vs": lambda: vs},
            current_t,
            i
        )

    postprocessor.finalize_all()


if __name__ == "__main__":
    setup_general_parameters()
    main()
