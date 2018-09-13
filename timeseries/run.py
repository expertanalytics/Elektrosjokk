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

import numba as nb

import warnings

import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

nbkwargs = {
    "nopython": True,
    "nogil": True,
    "cache": True,
    "fastmath": True
}

warnings.simplefilter("ignore", DeprecationWarning)

BASEDIR = "Documents"
DATAPATH = Path.home() / BASEDIR / "ECT-data"

df.set_log_level(100)

time_const = df.Constant(0)
dt = 1e-2
T = 30.0e3      # End time in [ms]
mesh = df.Mesh(str(DATAPATH / "meshes/bergenh18/merge.xml.gz"))
mesh.coordinates()[:] *= 10     # convert from mm to cm

# Boundary condition facet function
ff = df.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
ff.set_all(0)
df.CompiledSubDomain("x[0] > 5 && on_boundary").mark(ff, 11)

# White matter cell function
mf = df.MeshFunction("size_t", mesh, str(DATAPATH / "meshes/bergenh18/wm.xml.gz"))

# Load the anisotropy
Vv = df.TensorFunctionSpace(mesh, "CG", 1)
# fiber = df.Function(Vv, str(DATAPATH / "meshes/bergenh18/anisotropy.xml.gz"))
M_i = df.Function(Vv, str(DATAPATH / "meshes/bergenh18/intraanisotropy.xml.gz"))
M_e = df.Function(Vv, str(DATAPATH / "meshes/bergenh18/extraanisotropy.xml.gz"))


# def get_anisotropy(fiber, iso_value, k=1):
#     """Following Lee-2012 p6.""" 
#     ql = iso_value*k**(2./3)
#     qt = iso_value*k**(-1./3)
#     return fiber*df.diag(df.as_vector([ql, qt, qt]))*fiber.T


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:, :]), **nbkwargs)
def compute_point_weight(x, points_array):
    """Compute the normalised square distance between `x` and each point in `point_list`."""
    square_dist_matrix = np.power(x - points_array, 2).sum(axis=-1)
    scaled_point_weight = square_dist_matrix/square_dist_matrix.sum()
    return scaled_point_weight


@nb.jit( nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), **nbkwargs)
def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (x - x0)*(y1 - y0)/(x1 - x0)


@nb.jit(
    nb.float64(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:, :], nb.float64),
    **nbkwargs
)
def computed_value(x, channel_points, time_array, time_values, time):
    idx = np.searchsorted(time_array, time)

    val = 0
    factors = compute_point_weight(x, channel_points)
    for i in range(len(channel_points)):
        val += factors[i]*linear_interpolation(
            time,
            time_array[idx - 1],
            time_array[idx],
            time_values[i, idx - 1],
            time_values[i, idx]
        )
    return val


if rank == 0:
    # Load the EEG data
    start_idx = 1663389
    stop_idx = start_idx + 300000
    time_series = pd.read_pickle(DATAPATH / "zhi/EEG_signal.xz").values[:, start_idx:stop_idx]

    with open(DATAPATH / "zhi/channel.pos", "r") as channel_pos_handle:
        channel_points = [
            np.fromiter(
                # NB! coordinates in cm
                map(lambda x: float(x), ch[2:]),  # Cast each coordinate to float
                dtype=np.float64
            ) for ch in map(                         # Split all lines in ','
                lambda x: x.split(","),
                channel_pos_handle.readlines()
            )
        ]

    N = 100
    print("Only using {} channels".format(N))
    channel_points = np.array(channel_points[:N]).astype(np.float_)
    # Sample rate in 5kHz -- >  now in ms
    time_array = np.linspace(0, time_series.shape[1]/5, time_series.shape[1])
    time_series = time_series[:len(channel_points), :]
else:
    time_series = None
    channel_points = None
    time_array = None

# Broadcast the arrays. Only do file IO on one process
channel_points = comm.bcast(channel_points, root=0)
time_array = comm.bcast(time_array, root=0)
time_series = comm.bcast(time_series, root=0)


class MyExpression(df.Expression):
    """Set value to the sum of time series scaled by a normalised distance."""
    def __init__(
                self,
                time,
                point_list,
                time_array,
                time_values,
                **kwargs
    ):
        """Linear interpolation of each time series."""
        # self.time_func_list = interpolated_list
        self.time_axis = time_array
        self.time_values = time_values
        self.time = time                # The current time
        self.point_list = point_list    # The point corresponding to each time series

    def eval(self, value, x):
        """Scale the value of the time series with a normalised weight."""
        val = computed_value(
            x.astype(np.float64),
            self.point_list,
            self.time_axis,
            self.time_values,
            self.time(0)
        )
        # The EEG signals are in muV
        value[0] = val*1e-3     # Convert to mV

    def value_type(self):
        return (1,)


my_expr = MyExpression(
    time_const,
    channel_points,
    time_array,
    time_series,
    degree=1
)


model = Wei()
brain = CardiacModel(
    mesh,
    time_const,
    M_i=M_i,
    M_e=M_e,
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
ps["BidomainSolver"]["Chi"] = 1.26e3      # 1/cm -- Dougherty 2015
ps["BidomainSolver"]["Cm"] = 1.0          # muF/cm^2 -- Dougherty 2015

df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
df.parameters["form_compiler"]["cpp_optimize_flags"] = flags

# parameters.form_compiler.quadrature_degree = 1
solver = SplittingSolver(brain, params=ps)

vs_, *_ = solver.solution_fields()
REFERENCE_SOLUTION = pd.read_pickle(DATAPATH / "initial_conditions/REFERENCE_SOLUTION.xz").values
uniform_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="fire")      # flat

brain.cell_models().set_initial_conditions(**uniform_ic)
vs_.assign(model.initial_conditions())

field_spec = FieldSpec(save_as="xdmf", stride_timestep=10)

outpath =  Path.home() / "out/bergen_casedir"
pp_spec = PostProcessorSpec(casedir=outpath)

saver = Saver(pp_spec)
saver.store_mesh(mesh, facet_domains=ff)
saver.add_field(Field("v", field_spec))
saver.add_field(Field("u", field_spec))


point_array = np.array([
    [-46.56, -3.64,-17.89],
    [34.84, -3.64, 32.11],
    [1.18, -32.84, 1.59],
    [1.18, 67.28, 2.79]
])
ode_names = ("V", "m","h", "n", "NKo", "NKi", "NNao", "NNai", "NClo", "NCli", "vol", "O")
for name in ode_names:
    saver.add_field(PointField(name, field_spec, point_array))

theta = ps["theta"]

tick = time.clock()
for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
    norm = vs.vector().norm('l2')
    print("Solving ({t0}, {t1}) -----> {norm}".format(t0=t0, t1=t1, norm=norm))
    print()
    current_t = t0 + theta*(t1 - t0)
    v, u, *_ = vur.split(deepcopy=True)
    sol = vs.split(deepcopy=True)

    update_dict = {"v": v, "u": u}
    update_dict.update({name: func for name, func in zip(ode_names, sol)})

    saver.update(
        time_const,
        i,
        update_dict
    )
saver.close()
# print(f"Time: {time.clock() - tick}")
