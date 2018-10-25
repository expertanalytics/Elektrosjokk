import dolfin as df
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from xalbrain import SingleCellSolver
from xalbrain.cellmodels import Wei

import seaborn as sns


# For computing faster
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
df.parameters["form_compiler"]["cpp_optimize_flags"] = flags


def get_model() -> Wei:
    # set up model
    model_params = Wei.default_parameters()
    model = Wei(params=model_params)
    return model


def solve(model, dt, interval):
    # Simulation time
    time = df.Constant(0.0)

    # Set up solver
    solver_params = SingleCellSolver.default_parameters()
    solver = SingleCellSolver(model, time, params=solver_params)

    # Assign initial conditions
    vs_, vs = solver.solution_fields()
    vs_.assign(model.initial_conditions())


    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for ((t0, t1), vs) in solutions:
        print("Current time: {:g}".format(t1))
        times.append(t1)
        values.append(vs.vector().array())
    return np.asarray(times), np.asarray(values)


def plot_ode(time, value, label, dt=0.1):
    N = int(1 / dt)
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("ODE plot", size=52)

    ax = fig.add_subplot(111)
    # ax.grid(True)  # ???

    ax.plot(time[::N], value[::N], label=label, linewidth=2)
    ax.legend(fontsize=24)
    ax.set_xlabel("Time ms")
    fig.savefig(f"plots/{label}.png")
    plt.close(fig)



if __name__ == "__main__":
    from pathlib import Path
    df_path = Path("ode_solution.xz")

    # dt = 0.01
    dt = 0.1
    interval = (0.0, 120000.0)

    model = get_model()

    if df_path.exists():
        dataframe = pd.read_pickle(str(df_path))
    else:
        times, values = solve(model, dt, interval)
        dataframe = pd.DataFrame(np.concatenate((times[:,None], values), axis=1))
        dataframe.to_pickle(str(df_path))

    times = dataframe.values[:, 0]
    values = dataframe.values[:, 1:].T

    # times, values = solve(model, dt, interval)
    sns.set()
    # plot_ode(times[7000000:7010000], values[0][7000000:7010000], "V")
    # plot_ode(times[700000:701000], values[0][700000:701000], "V")

    labels = ("V", "m", "h", "n", "NKo", "NKi", "NNao", "NNai","NClo", "NCli", "vol", "O")
    for value, label in zip(values, labels):
        plot_ode(times, value, label)
