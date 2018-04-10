import dolfin
import numpy as np

from ect.postprocessing import (
    read_hdf5_mesh,
    read_hdf5_functions,
)

from ect.odesolvers import Wei, SPEC

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import seaborn
seaborn.set()


def read_mesh(filename):
    return read_hdf5_mesh(filename)


def read_functions(mesh):
    fields = [f"vs{t}" for t in range(1000)]
    # fields = [f"v{t}" for t in range(10)]
    func_generator = read_hdf5_functions(mesh, "uniform_casedir/vs/vs.hdf5", fields)
    # func_generator = read_hdf5_functions(mesh, "test/v/v.hdf5", fields)
    for v in func_generator:
        # yield v
        yield v.split(deepcopy=False)

def find_idx(coordinates, point):
    coordinates = coordinates.copy()
    coordinates -= np.asarray(point)
    np.abs(coordinates, out=coordinates)
    return np.argmin(np.sum(coordinates, axis=1))


if __name__ == "__main__":
    mesh = read_mesh("uniform_casedir/mesh.hdf5")
    point = (0.5, 0.5)
    # coordinates = mesh.coordinates()
    # idx = find_idx(coordinates, (0.5, 0.5))

    from itertools import chain
    
    # for f in read_functions(mesh):
        # print(f(0.5, 0.5))

    try:
        pde_values = np.load("pde_values.npy")
    except Exception as e:
        pde_values = np.asarray([
            list(map(lambda x: x(*point), v)) for v in read_functions(mesh)
        ])
        np.save("pde_values.npy", pde_values)
    ic = pde_values[0]

    dt = 1e-3
    T = 1e0

    import numba
    Wei_compiled = numba.jitclass(SPEC)(Wei)
    solver = Wei_compiled(ic, T, dt)
    solver.solve()

    t_array = solver.t_array
    ode = solver.solution
    pde = pde_values

    fig, ax = plt.subplots()
    ax.set_title("Chaotic Initial Conditions", fontsize=36)
    ax.plot(t_array, ode[:, 0], label="ode", linewidth=2)
    ax.plot(t_array, pde[:, 0], label="pde", linewidth=2)
    ax.set_xlabel("Time (ms)", fontsize=18)
    ax.set_ylabel("Transmembrane potential (mV)", fontsize=18)
    plt.legend(fontsize=16)
    fig.savefig("uniform.png")
