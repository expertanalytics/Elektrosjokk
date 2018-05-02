import dolfin

import numpy as np
import xalbrain as xb

from ect.postprocessing import (
    read_hdf5_mesh,
    read_hdf5_functions,
)

from ect.odesolvers import fenics_ode_solver

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
    func_generator = read_hdf5_functions(mesh, "chaotic_casedir/vs/vs.hdf5", fields)
    # func_generator = read_hdf5_functions(mesh, "test/v/v.hdf5", fields)
    for v in func_generator:
        # yield v
        yield v.split(deepcopy=False)


def find_idx(coordinates, point):
    coordinates = coordinates.copy()
    coordinates -= np.asarray(point)
    np.abs(coordinates, out=coordinates)
    return np.argmin(np.sum(coordinates, axis=1))


def get_values(mesh, point, functions):
    point_identifier = str(point).replace(".", "").replace(", ", "")
    name = f"pde_values{point_identifier}.npy"
    try:
        pde_values = np.load(name)
    except Exception as e:
        pde_values = np.asarray([
            list(map(lambda x: x(*point), v)) for v in read_functions(mesh)
        ])
        np.save("pde_values.npy", pde_values)
    ic = pde_values[0]

    dt = 1e-3
    T = 1e0

    model = xb.cellmodels.Wei()
    ic_dict = model.default_initial_conditions()
    for i, key in enumerate(ic_dict):
        ic_dict[key] = ic[i]
    
    t_array = []
    ode_values = []
    for t, v in fenics_ode_solver(model, dt, (0, T), ic=ic_dict):
        t_array.append(t)
        ode_values.append(v)
    
    ode_values = np.asarray(ode_values)
    t_array = np.asarray(t_array)
    return t_array, pde_values, ode_values



if __name__ == "__main__":
    mesh = read_mesh("chaotic_casedir/mesh.hdf5")
    functions = read_functions(mesh)
    # point = (0.5, 0.5)
    for i, point in enumerate([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)]):
        t_array, pde, ode = get_values(mesh, point, functions)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"point: {point}", fontsize=36)
        ax.plot(t_array, ode[:, 0], label="ode")
        ax.plot(t_array, pde[:, 0], label="pde")
        ax.set_xlabel("Time (ms)", fontsize=18)
        ax.set_ylabel("Transmembrane potential (mV)", fontsize=18)
        plt.legend(fontsize=16)
        fig.savefig(f"chaotic{i}.png")
