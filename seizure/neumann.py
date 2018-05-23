"""Run case with neumann BC."""

import numpy as np

from xalbrain import (
    UnitSquareMesh,
    CardiacModel,
    SplittingSolver,
    Constant,
    Expression,
    SubDomain,
    MeshFunction,
)

from xalbrain.cellmodels import Wei

from cbcpost import (
    PostProcessor,
    Field,
    SolutionField,
)

from ect.specs import SolutionFieldSpec

from ect.utilities import wei_uniform_ic


time = Constant(0)
T = 0.1e1
dt = 2e-3
mesh = UnitSquareMesh(100, 100)

lower_ect_current = Expression(
    "std::abs(cos(2*pi*70e-3*t))*(t < t0) > 0.5 ? 800 : 0",
    t=time,
    t0=T,
    degree=1
)

upper_ect_current = Expression(
    "std::abs(cos(2*pi*70e-3*t))*(t < t0) > 0.5 ? -800 : 0",
    t=time,
    t0=T,
    degree=1
)

class UpperBox(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] > 0.55) and (x[1] < 0.9) and (x[0] > 0.1) and (x[0] < 0.9)


class LowerBox(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] > 0.1) and (x[1] < 0.45) and (x[0] > 0.1) and (x[0] < 0.9)

mf = MeshFunction("size_t", mesh, 2)        # NB! 2 == CellFunction
mf.set_all(0)
LowerBox().mark(mf, 1)
UpperBox().mark(mf, 2)

upper_expr_str = ",".join((
    "(x[1] - 0.55)*(1*(x[0] > 0.5) - 1*(x[0] < 0.5))",
    "1 - 4*(x[0] - 0.5)*(x[0] - 0.5)"
))

lower_expr_str = ",".join((
    "(0.45 - x[1])*(1*(x[0] > 0.5) - 1*(x[0] < 0.5))",
    "4*(x[0] - 0.5)*(x[0] - 0.5) - 1"
))

upper_expression = Expression(upper_expr_str, degree=1)
lower_expression = Expression(lower_expr_str, degree=1)

class UpperCorner(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.5 and x[1] > 0.5 and on_boundary


class LowerCorner(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 and x[1] < 0.5 and on_boundary

ff = MeshFunction("size_t", mesh, 1)
ff.set_all(0)
LowerCorner().mark(ff, 1)
UpperCorner().mark(ff, 2)

foo_expr = Expression("x[0] + x[1], x[0] - x[1]", degree=1)

model = Wei()
brain = CardiacModel(
    mesh,
    time,
    M_i={2: 0.1*Expression("x[0] + x[1], x[0] - x[1]", degree=1), 1: Constant(0.1), 0: Constant(0)},
    M_e={2: 0.3*Expression("x[0] + x[1], x[0] - x[1]", degree=1), 1: Constant(0.3), 0: Constant(1.6)},
    cell_models=model,
    ect_current={
        1: 1.6*lower_ect_current,       # NB! Remeber to include the CSF conductivity
        2: 1.6*upper_ect_current
    },
    cell_domains=mf,
    facet_domains=ff
)

ps = SplittingSolver.default_parameters()
ps["pde_solver"] = "bidomain"
ps["BidomainSolver"]["linear_solver_type"] = "direct"
ps["BidomainSolver"]["use_avg_u_constraint"] = True
ps["theta"] = 0.5
solver = SplittingSolver(brain, params=ps)

vs_, *_ = solver.solution_fields()
REFERENCE_SOLUTION = np.load("REFERENCE_SOLUTION.npy")
uniform_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="flat")
brain.cell_models().set_initial_conditions(**uniform_ic)

field_spec = SolutionFieldSpec(stride_timestep=1)
postprocessor = PostProcessor(dict(casedir="really_long_casedir", clean_casedir=True))
postprocessor.store_mesh(brain.mesh)
postprocessor.add_field(SolutionField("v", field_spec._asdict()))
postprocessor.add_field(SolutionField("u", field_spec._asdict()))
# postprocessor.add_field(SolutionField("vs", field_spec._asdict()))
theta = ps["theta"]

REFERENCE_SOLUTION = np.load("REFERENCE_SOLUTION.npy")
uniform_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="flat")
brain.cell_models().set_initial_conditions(**uniform_ic)
vs_.assign(model.initial_conditions())

for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
    print(f"Solving ({t0}, {t1})")
    # functions = vs.split()
    current_t = t0 + theta*(t1 - t0)
    v, u, _ = vur.split(deepcopy=True)

    postprocessor.update_all(
        {"v": lambda: v, "u": lambda: u},
        current_t,
        i
    )
postprocessor.finalize_all()
