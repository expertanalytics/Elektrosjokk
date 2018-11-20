import dolfin as df
import numpy as np

from scipy.spatial import cKDTree
from pathlib import Path

from eegutils import read_zhe_eeg
from postplot import mplot_function

from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt

font = {
    "family": "DejaVu Sans",
    "weight": "bold",
    "size": 20,
}

tick= {"labelsize": 10}

mpl.rc("font", **font)
mpl.rc("xtick", **tick)
mpl.rc("ytick", **tick)


def get_eeg_value() -> List[float]:
    # Rather than use a class, I can use the send/recv feature of generators

    all_channels_seizure = read_zhe_eeg(
        start=4078000,
        stop=4902000,
        full=True
    )
    return all_channels_seizure


def get_positions(ect_dir_path: str="Documents/ECT-data") -> np.ndarray:
    """Read the electrode positions, and return a 2d array of positions.

    Arguments:
        ect_dir_path: Path to the ECT-data directory.
    """
    _ect_dir_path = Path(ect_dir_path)
    data = np.loadtxt(
        Path.home() / _ect_dir_path / "zhi/channel.pos",
        delimiter=",",
        usecols=(2, 3, 4)
    )
    x, y, _ =  data.T
    return x, y


my_points = list(zip(*get_positions()))
mesh = df.Mesh("gmsh_meshing/protected_points.xml")
coordinates = mesh.coordinates()[:, :2]

tree = cKDTree(coordinates)
distances, mesh_point_indices = tree.query(my_points)
assert (distances < 1e-5).all(), "Problems with mesh point look up"

function_space = df.FunctionSpace(mesh, "CG", 1)

u = df.TrialFunction(function_space)
v = df.TestFunction(function_space)
solution_function = df.Function(function_space)

lhs = df.inner(df.grad(u), df.grad(v))*df.dx
bc = df.DirichletBC(function_space, df.Constant(0), df.DomainBoundary())

func = df.Function(function_space)

vd_map = df.vertex_to_dof_map(function_space)
dv_map = df.dof_to_vertex_map(function_space)

coeffs = np.zeros(shape=vd_map.size)


A = df.assemble(lhs)

solver = df.KrylovSolver("cg", "amg")
solver.set_operator(A)

# maxlist = []
# minlist = []

# EEG values
all_eeg_channels = get_eeg_value()
for i in range(1000):
    print(i)
    time = 5*i    # Sampling frequenct of 5 kHz
    z = all_eeg_channels[:, time]

    # FIXME: This is proof of concept. Find the defective channel indices
    coeffs[vd_map[mesh_point_indices]] = z[:vd_map[mesh_point_indices].size]
    func.vector().set_local(coeffs)

    rhs = func*v*df.dx
    bb = df.assemble(rhs)
    bc.apply(A, bb)

    solver.solve(solution_function.vector(), bb)
    assert solution_function.vector().norm("l2")/solution_function.vector().size() < 1e9

    # maxlist.append(solution_function.vector().get_local().max())
    # minlist.append(solution_function.vector().get_local().min())

    fig, ax = mplot_function(
        solution_function,
        vmin=-750,
        vmax=750,
        colourbar=True,
        colourbar_label="$\mu V$"
    )

    ax.set_xlabel("cm")
    ax.set_ylabel("cm")
    ax.set_title("$u^*$")
    fig.savefig("figures/eeg{:04d}.png".format(i))
    plt.close(fig)

# print(max(maxlist))
# print(min(minlist))
