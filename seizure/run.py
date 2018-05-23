"upper_anisotropy""Test case for xalpost."""

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
    Function,
    project,
    FunctionSpace,
    VectorFunctionSpace,
    as_matrix,
    as_vector,
    diag,
)

from xalbrain.cellmodels import Wei

from postutils import wei_uniform_ic

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
T = 5e1
dt = 1e-2
mesh = UnitSquareMesh(50, 50)


lower_ect_current = Expression(
    "std::abs(sin(2*pi*f*t))*(t < t0) > sin(2*pi*width/2) ? 300 : 0",
    t=time,
    t0=T,
    width=1.0,
    f=70e-1,        # default: 70e-3
    degree=1
)

upper_ect_current = Expression(
    "std::abs(sin(2*pi*f*t))*(t < t0) > sin(2*pi*width/2) ? -300 : 0",
    t=time,
    t0=T,
    width=1.0,
    f=70e-3,
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


class Xdir(Expression):
    def eval(self, value, x):
        if x[1] >= 0.5:
            value[0] = x[1]*x[0]
            value[1] = 1.0 - x[0]
        else:
            value[0] = -x[1]*x[0]
            value[1] = x[0] - 1

    def value_shape(self):
        return (2,)


class Ydir(Expression):
    def eval(self, value, x):
        if x[1] >= 0.5:
            value[0] = 1.0 - x[0]
            value[1] = -x[1]*x[0]
        else:
            value[0] = -(x[0] - 1)
            value[1] = x[1]*x[0]

    def value_shape(self):
        return (2,)


Vv = VectorFunctionSpace(mesh, "CG", 1)
fiber = project(Xdir(degree=1), Vv)
transverse = project(Ydir(degree=1), Vv)

A = as_matrix([
    [fiber[0], transverse[0]],
    [fiber[1], transverse[1]]
])

upper_intra_anisotropy = A*diag(as_vector([Constant(10), Constant(0.1)]))*A.T
upper_extra_anisotropy = A*diag(as_vector([Constant(27), Constant(2.7)]))*A.T
lower_intra_anisotropy = A*diag(as_vector([Constant(10), Constant(0.1)]))*A.T
lower_extra_anisotropy = A*diag(as_vector([Constant(27), Constant(2.7)]))*A.T

model = Wei()
brain = CardiacModel(
    mesh,
    time,
    M_i={2: upper_intra_anisotropy, 1: lower_intra_anisotropy, 0: Constant(0)},
    M_e={2: upper_extra_anisotropy, 1: lower_extra_anisotropy, 0: Constant(165.4)},
    cell_models=model,
    ect_current={
        1: Constant(165.4)*lower_ect_current,       # NB! Remeber to include the CSF conductivity
        2: Constant(165.4)*upper_ect_current
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

field_spec = FieldSpec()
pp_spec = PostProcessorSpec(casedir="test_pp_casedir")

saver = Saver(pp_spec)
saver.store_mesh(mesh, cell_domains=mf, facet_domains=ff)
saver.add_field(Field("v", field_spec))
saver.add_field(Field("u", field_spec))
points = [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)]
saver.add_field(PointField("NKo-point", field_spec, np.asarray(points)))
saver.add_field(PointField("NNao-point", field_spec, np.asarray(points)))
saver.add_field(PointField("NClo-point", field_spec, np.asarray(points)))
saver.add_field(PointField("Voli-point", field_spec, np.asarray(points)))

theta = ps["theta"]
for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
    print(f"Solving ({t0}, {t1}) -----> {vs.vector().norm('l2')}")
    current_t = t0 + theta*(t1 - t0)
    # functions = vs.split()
    v, u, *_ = vur.split(deepcopy=True)
    sol = vs.split(deepcopy=True)
    print()

    saver.update(
        time,
        i,
        {
            "v": v,
            "u": u,
            "NKo-point": sol[4],
            "NNao-point": sol[6],
            "NClo-point": sol[8],
            "Voli-point": sol[10]
        }
    )
saver.finalise()
