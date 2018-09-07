import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

import dolfin as df

from mpi4py import MPI

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

from pathlib import Path

import warnings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

warnings.simplefilter("ignore", DeprecationWarning)

BASEDIR = "shared"
DATAPATH = Path.home() / BASEDIR / "ECT-data"

df.set_log_level(100)

time = df.Constant(0)
dt = 1e-2
T = 50.0      # End time in [ms]
mesh = df.Mesh(str(DATAPATH / "meshes/bergenh18/merge.xml.gz"))

# Boundary condition facet function
ff = df.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
ff.set_all(0)
df.CompiledSubDomain("x[2] > 20 && on_boundary").mark(ff, 11)

# White matter cell function
mf = df.MeshFunction("size_t", mesh, str(DATAPATH / "meshes/bergenh18/wm.xml.gz"))

# Load the anisotropy
Vv = df.TensorFunctionSpace(mesh, "CG", 1)
fiber = df.Function(Vv, str(DATAPATH / "meshes/bergenh18/anisotropy.xml.gz"))


def get_anisotropy(fiber, iso_value):
    return fiber*df.diag(df.as_vector([iso_value]*3))*fiber.T


def compute_point_weight(x, points_array):
    """Compute the normalised square distance between `x` and each point in `point_list`."""
    # square_dist_matrix = np.sqrt(np.power(x - points_array, 2).sum(axis=-1))
    # Use distance squared, not distance
    square_dist_matrix = np.power(x - points_array, 2).sum(axis=-1)
    scaled_point_weight = square_dist_matrix/square_dist_matrix.sum()
    return scaled_point_weight


if rank == 0:
    # Load the EEG data
    start_idx = 1663389
    stop_idx = start_idx + 300000
    time_series = pd.read_pickle(DATAPATH / "zhi/EEG_signal.xz").values[:, start_idx:stop_idx]

    with open(DATAPATH / "zhi/channel.pos", "r") as channel_pos_handle:
        channel_points = [
            np.fromiter(
                map(lambda x: 10*float(x), ch[2:]),  # Cast each coordinate to float and convert to mm
                dtype=np.float64
            ) for ch in map(                         # Split all lines in ','
                lambda x: x.split(","),
                channel_pos_handle.readlines()
            )
        ]

    print("Only using 10 channels")
    channel_points = channel_points[:10]
    # Sample rate in 5kHz -- >  now in ms
    time_array = np.linspace(0, time_series.shape[1]/5, time_series.shape[1])
    interpolated_list = [
        interp1d(time_array, ts) for ts in time_series[:len(channel_points), :]
    ]
else:
    interpolated_list = None
    channel_points = None

# Broadcast the interpolation
interpolated_list = comm.bcast(interpolated_list, root=0)
channel_points = comm.bcast(channel_points, root=0)


class MyExpression(df.Expression):
    """Set value to the sum of time series scaled by a normalised distance."""

    def __init__(
            self,
            time,
            point_list,
            interpolated_list,
            **kwargs
    ):
        """Linear interpolation of each time series."""
        self.time_func_list = interpolated_list
        self.time = time                # The current time
        self.point_list = point_list    # The point corresponding to each time series

    def eval(self, value, x):
        """Scale the value of the time series with a normalised weight."""

        val = sum([
            factor*tf(self.time(0)) for factor, tf in zip(
                compute_point_weight(x, self.point_list),
                self.time_func_list,
            )
        ])
        # The EEG signals are in muV
        value[0] = val*1e-3     # Convert to mV

    def value_type(self):
        return (1,)


my_expr = MyExpression(
    time,
    channel_points,
    interpolated_list,
    degree=1
)

model = Wei()
brain = CardiacModel(
    mesh,
    time,
    # Units now in mS/mm
    M_i={0: get_anisotropy(fiber, 0.1), 11: get_anisotropy(fiber, 0.1)
    },
    M_e={
        0: get_anisotropy(fiber, 0.276), 11: get_anisotropy(fiber, 0.126)
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
ps["BidomainSolver"]["Chi"] = 1.26e2     # 1/mm -- Dougherty 2015
ps["BidomainSolver"]["Cm"] = 1e-10         # F/mm^2 -- Dougherty 2015

# parameters.form_compiler.quadrature_degree = 1
solver = SplittingSolver(brain, params=ps)

vs_, *_ = solver.solution_fields()
REFERENCE_SOLUTION = pd.read_pickle(DATAPATH / "initial_conditions/REFERENCE_SOLUTION.xz").values
uniform_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="flat")

brain.cell_models().set_initial_conditions(**uniform_ic)
vs_.assign(model.initial_conditions())

field_spec = FieldSpec(save_as="xdmf", stride_timestep=10)

outpath =  Path.home() / "out/bergen_casedir"
pp_spec = PostProcessorSpec(casedir=outpath)

saver = Saver(pp_spec)
saver.store_mesh(mesh, facet_domains=ff)
saver.add_field(Field("v", field_spec))
saver.add_field(Field("u", field_spec))

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
            "u": u
        }
    )
saver.close()
