import numpy as np

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

from dolfin import (
    Mesh,
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
    parameters,
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
mesh = Mesh("data/merge.xml.gz")

lower_ect_current = Expression(
    "std::abs(sin(2*pi*f*t))*(t < t0) > sin(2*pi*width/2) ? 300 : 0",
    t=time,
    t0=T,
    width=1.0,
    f=70e-0,        # default: 70e-3
    degree=1
)

upper_ect_current = Expression(
    "std::abs(sin(2*pi*f*t))*(t < t0) > sin(2*pi*width/2) ? -300 : 0",
    t=time,
    t0=T,
    width=1.0,
    f=70e-0,
    degree=1
)


class Front(SubDomain):
    def inside(self, x, on_boundary):
        sphere = (x[0] - 40.54)**2 + (x[1] - 73.28)**2 + (x[2] - 46.33)**2
        return sphere < 10.**2


class Lateral(SubDomain):
    def inside(self, x, on_boundary):
        sphere = (x[0] - 77.47)**2 + (x[1] - 13.80)**2 + (x[2] - 11.45)**2
        return sphere < 10.**2

ff = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
ff.set_all(0)
Front().mark(ff, 1)
Lateral().mark(ff, 2)

mf = MeshFunction("size_t", mesh, "data/merge_physical_region.xml")

Vv = VectorFunctionSpace(mesh, "CG", 1)
fiber = Function(Vv, "data/anisotropy.xml.gz")

A = as_matrix([
    [fiber[0]],
    [fiber[1]],
    [fiber[2]]
])

# Dicvide by 10 in the other direction. No Idea why
# intra_anisotropy = A*diag(as_vector([Constant(10), Constant(1.0)]))*A.T
# extra_anisotropy = A*diag(as_vector([Constant(27), Constant(2.7)]))*A.T
intra_anisotropy = Constant(1)
extra_anisotropy = Constant(1)

model = Wei()
brain = CardiacModel(
    mesh,
    time,
    M_i={1: intra_anisotropy, 2: Constant(0)},
    M_e={1: extra_anisotropy, 2: Constant(165.4)},
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
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["use_avg_u_constraint"] = True
ps["theta"] = 0.5

# parameters.form_compiler.quadrature_degree = 1

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
