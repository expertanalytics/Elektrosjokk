"""Test case for xalpost."""

import numpy as np

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

from dolfin import (
    UnitSquareMesh,
    Constant,
    Expression,
    SubDomain,
    MeshFunction,
)

from xalbrain.cellmodels import Wei

from ect.utilities import wei_uniform_ic

from post import Saver

from postfields import (
    Field,
    PointField, 
)

from postspec import (
    FieldSpec,
    PostProcessorSpec,
)


time = Constant(0)
T = 1e-1
dt = 2e-2
mesh = UnitSquareMesh(50, 50)

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
        1: Constant(1.6)*lower_ect_current,       # NB! Remeber to include the CSF conductivity
        2: Constant(1.6)*upper_ect_current
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
vs_.assign(model.initial_conditions())

# New postprocessor from here <------
field_spec = FieldSpec()
pp_spec = PostProcessorSpec(casedir="test_pp_casedir")

saver = Saver(pp_spec)
saver.store_mesh(mesh, cell_domains=mf, facet_domains=ff)
saver.add_field(Field("v", field_spec))
points = [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)]
saver.add_field(PointField("NKo-point", field_spec, np.asarray(points)))
saver.add_field(PointField("Nnao-point", field_spec, np.asarray(points)))
saver.add_field(PointField("NClo-point", field_spec, np.asarray(points)))

theta = ps["theta"]
for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
    print(f"Solving ({t0}, {t1})")
    current_t = t0 + theta*(t1 - t0)
    # functions = vs.split()
    # v, u, _ = vur.split(deepcopy=True)

    saver.update(time, i, {"v": v, "v-point": v})

saver.finalise()
