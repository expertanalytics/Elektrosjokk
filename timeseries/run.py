import numpy as np

from scipy.interpolate import interp1d

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

import dolfin as df

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

df.set_log_level(100)

time = df.Constant(0)
dt = 1e-2
T = 50.0      # End time in [ms]
mesh = df.Mesh("data/merge.xml.gz")

# Boundary condition facet function
ff = df.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
ff.set_all(0)
df.CompiledSubDomain("x[0] < 5 && on_boundary").mark(ff, 11)

# White matter cell function
mf = df.MeshFunction("size_t", mesh, "data/wm.xml.gz")

# Load the anisotropy
Vv = df.TensorFunctionSpace(mesh, "CG", 1)
fiber = df.Function(Vv, "data/anisotropy_correct.xml.gz")


def get_anisotropy(fiber, iso_value):
    return fiber*df.diag(df.as_vector([iso_value]*3))*fiber.T


def compute_point_weight(x, points_array):
    """Compute the normalised square distance between `x` and each point in `point_list`."""
    square_dist_matrix = np.sqrt(np.power(x - points_array, 2).sum(axis=-1))
    scaled_point_weight = square_dist_matrix/square_dist_matrix.sum()
    return scaled_point_weight


# Load the EEG data
start_idx = 1663389
time_series = np.load("data/EEG_signals.npy")[:, start_idx:]

# Sample rate in 5kHz -- >  now in ms
time_array = np.linspace(0, time_series.shape[1]/5, time_series.shape[1])
sample_points = np.array([(56, -43, 25)])   # Found by clicking randomly in Paraview


class MyExpression(df.Expression):
    """Set value to the sum of time series scaled by a normalised distance."""

    def __init__(
            self,
            time_axis,
            time_series_list,
            time,
            point_list,
            **kwargs
    ):
        """Linear interpolation of each time series."""
        print(np.max(time_series_list[0]))
        print(np.min(time_series_list[0]))
        self.time_func_list = [
            interp1d(time_axis, ts) for ts in time_series_list
        ]
        self.time = time                # The current time
        self.point_list = point_list    # The point corresponding to each time series

    def eval(self, value, x):
        """Scale the value of the time series with a normalised weight."""

        val = sum([factor*tf(self.time(0)) for factor, tf in zip(
                compute_point_weight(x, sample_points),
                self.time_func_list,
        )])

        import IPython
        IPython.embed()
        assert False
        # The EEG signals are in muV
        value[0] = val*1e-3     # mV

    def value_type(self):
        return (1,)



my_expr = MyExpression(
    time_array,
    time_series[:len(sample_points), :],
    time,
    sample_points,
    degree=1
)
# my_expr = df.Constant(1)

model = Wei()
brain = CardiacModel(
    mesh,
    time,
    # unsure about the units
    M_i={0: get_anisotropy(fiber, 0.1e3), 11: get_anisotropy(fiber, 0.1e3)
    },
    M_e={
        0: get_anisotropy(fiber, 0.276e3), 11: get_anisotropy(fiber, 0.126e3)
    },
    cell_models=model,
    facet_domains=ff,
    cell_domains=mf,
    dirichlet_bc=[(my_expr, 11)]
)

ps = SplittingSolver.default_parameters()
ps["pde_solver"] = "bidomain"
ps["theta"] = 0.5
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["use_avg_u_constraint"] = False

# Still unsure about units
ps["BidomainSolver"]["Chi"] = 1.6e2     # 1/mm -- Dougherty 2015
ps["BidomainSolver"]["Cm"] = 1e1         # mF/mm^2 -- Wei

# parameters.form_compiler.quadrature_degree = 1
solver = SplittingSolver(brain, params=ps)

vs_, *_ = solver.solution_fields()
REFERENCE_SOLUTION = np.load("REFERENCE_SOLUTION.npy")
uniform_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="flat")

print(uniform_ic)
1/0

brain.cell_models().set_initial_conditions(**uniform_ic)
vs_.assign(model.initial_conditions())

field_spec = FieldSpec()
pp_spec = PostProcessorSpec(casedir="test_pp_casedir")

saver = Saver(pp_spec)
saver.store_mesh(mesh, facet_domains=ff)
saver.add_field(Field("v", field_spec))
saver.add_field(Field("u", field_spec))
points = [(1.18, -6.79, 19.72)]
saver.add_field(PointField("NKo-point", field_spec, np.asarray(points)))
saver.add_field(PointField("NNao-point", field_spec, np.asarray(points)))
saver.add_field(PointField("NClo-point", field_spec, np.asarray(points)))
saver.add_field(PointField("Voli-point", field_spec, np.asarray(points)))

theta = ps["theta"]
for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
    print(f"Solving ({t0}, {t1}) -----> {vs.vector().norm('l2')}")
    current_t = t0 + theta*(t1 - t0)
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
saver.close()
