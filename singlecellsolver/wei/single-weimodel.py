"""Test the Wei model"""

import math
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from wei_utils import (
    get_uniform_ic,
)

import numpy as np

import time as systime

from xalbrain import (
    SingleCellSolver,
    BasicSingleCellSolver,
    Constant,
    parameters,
    Wei,
)

# For computing faster
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags


def main(initial_conditions: np.ndarray = None):
    """Solve a single cell model on some time frame."""

    model = Wei()
    time = Constant(0.0)
    # params = BasicSingleCellSolver.default_parameters()
    params = SingleCellSolver.default_parameters()

    params["scheme"] = "RK4"
    # solver = BasicSingleCellSolver(model, time, params)
    solver = SingleCellSolver(model, time, params)

    # Assign initial conditions
    vs_, vs = solver.solution_fields()
    # vs_.assign(model.initial_conditions())
    model.set_initial_conditions(**get_uniform_ic("flat"))
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    # N = 25*(14000 + 3000)

    dt = 1e-3
    # T = 1500 
    # N = T/dt + 1
    # assert False, N
    # interval = (0, N)

    # N = 6000*0.02
    # dt = 0.02
    # interval = (0.0, N)
    T = 1000.0
    interval = (0.0, T)

    start = systime.clock() 
    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for i, ((t0, t1), vs) in enumerate(solutions):        # TODO: Unpacking from 3.6?
        times.append(t1)
        if i % 1 == 0:
            print(i, vs.vector().norm("l2"))
        values.append(vs.vector().get_local())
        if i % 1 == 0:
            print(f"step: {i}", flush=True)
    print("Time to solve: {}".format(systime.clock() - start))

    return times, values


def plot_all(times, values, odir="figures"):
    # points = data_dict["points"]
    # time = tuple(filter(lambda x: x != "points", data_dict.keys()))

    # function_dict = {key: [] for key in data_dict[time[0]].keys()}
    # for t in time:
    #     for key in data_dict[t]:
    #         function_dict[key].append(np.array(data_dict[t][key]))
    # np.vstack(function_dict["vs-0"])
    # plot_dict = {key: np.vstack(value) for key, value in function_dict.items()}

    data_matrix = np.array(values)[:, :12]
    time = np.array(times)

    vol = 1.4368e-15
    voli = data_matrix[:, 10]
    beta0 = 7
    volo = (1 + 1/beta0)*vol - voli

    data_matrix[:, 4] /= volo
    data_matrix[:, 5] /= voli
    data_matrix[:, 6] /= volo
    data_matrix[:, 7] /= voli
    data_matrix[:, 8] /= volo
    data_matrix[:, 9] /= voli
    data_matrix[:, 10] /= volo

    titles = (
        r"Transmembrane potential", r"Voltage Gate (m)", r"Voltage Gate (n)", r"Voltage Gate (h)",
        r"Extracellular Potassium $[K^+]$", r"Intracellular Potessium $[K^+]$",
        r"Extracellular Sodium $[Na^+]$", r"Intracellular Sodium $[Na^+]$",
        r"Exctracellular Chlorine $[Cl^-]$", r"Intracellular Chlorine $[CL^-]$",
        r"Ratio of intracellular to extracellular volume", r"Extracellular Oxygen $[O_2]$"
        )
    ylabels = ("mV", "mV", "mV", "mV", "mol", "mol", "mol", "mol", "mol", "mol",
               r"$Vol_i/Vol_e$", "mol")
    names = {
        "V": "vs-0",
        "m": "vs-1",
        "h": "vs-2",
        "n": "vs-3",
        "Ko": "vs-4",
        "Ki": "vs-5",
        "Nao": "vs-6",
        "Nai": "vs-7",
        "Clo": "vs-8",
        "Cli": "vs-9",
        "beta": "vs-10",
        "O": "vs-11"
    }

    for i, (name, _ylabel, title) in enumerate(zip(names, ylabels, titles)):
        # fig, ax1 = plt.subplots(1, figsize=(8, 8))
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(111)
        fig.suptitle(title, size=52)
        data = data_matrix[:, i]        # axis 1 is over sample points

        ax1.plot(time, data, label=f"{name}", linewidth=4)
        ax1.set_ylabel(_ylabel, size=48)
        ax1.set_xlabel("Time (s)", size=48)
        ax1.grid()

        x0,x1 = ax1.get_xlim()
        y0,y1 = ax1.get_ylim()
        ax1.set_aspect((x1-x0)/(y1-y0))

        ax1.tick_params(axis="both", which="major", labelsize=28)
        ax1.tick_params(axis="both", which="minor", labelsize=28)

        tx = ax1.xaxis.get_offset_text()
        tx.set_fontsize(28)
        ty = ax1.yaxis.get_offset_text()
        ty.set_fontsize(28)

        # ax1.set_ylim(ylim)
        # ax1.legend(loc="best", prop={"size": 24})
        fig.savefig(f"{odir}/{name}_pde.png")
        plt.close(fig)


def plot_results(times, values, show=True):
    "Plot the evolution of each variable versus time."
    variables = list(zip(*values))

    # print(variables[1])
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(times, variables[0])
    print(len(times), len(variables[1]))
    ax1 = fig.add_subplot(211)
    print("time: ", max(times))
    ax1.plot(times, variables[0])
    ax2 = fig.add_subplot(212)
    ax2.plot(times, variables[2])
    print("time: ", max(times))
    fig.savefig("foo.pdf")


if __name__ == "__main__":
    #ic = np.load("initial_condition.npy")
    #s_ic = np.load("s_ic.npy")
    #V_ic = np.load("V_ic.npy")
    #ic = np.concatenate(((V_ic,), s_ic))
    #assert False, ic

    tic = systime.time()
    times, values = main()
    #times, values = main(ic[:12])
    #times, values = main()
    print(systime.time() - tic)
    np.save("solution.npy", values)
    # np.save("initial_condition.npy", values[-1])
    # plot_results(times, values, show=False)
    plot_all(times, values)
