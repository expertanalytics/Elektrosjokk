import dolfin as df
import numpy as np
import pandas as pd
import xalbrain as xb
import seaborn as sns
import matplotlib.pyplot as plt

from xalbrain.cellmodels import Cressman

from typing import (
    Tuple,
)


def plot_ode(time, value, label):
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("ODE plot", size=52)

    ax = fig.add_subplot(111)

    ax.plot(time, value, label=label, linewidth=2)
    ax.legend(fontsize=24)
    ax.set_xlabel("Time ms")
    fig.savefig(f"plots/{label}.png")
    plt.close(fig)


def get_model() -> Cressman:
    """Set up the cell model with parameters."""
    model_params = Cressman.default_parameters()
    model_params["Koinf"] = 8
    model = Cressman(params=model_params)
    return model


def solve(model: xb.CardiacCellModel, dt: float, interval: Tuple[float, float]) -> None:
    time = df.Constant(0)

    # Set up solver
    solver_params = xb.BasicSingleCellSolver.default_parameters()
    # solver_params["scheme"] = "GRL1"
    solver_params["V_polynomial_family"] = "CG"
    solver_params["V_polynomial_degree"] = 1
    solver = xb.BasicSingleCellSolver(model, time, params=solver_params)

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



if __name__ == "__main__":
    df.set_log_level(100)

    from pathlib import Path
    df_path = Path("cressman_solution.xz")

    dt = 0.01
    interval = (0.0, 1e2)
    model = get_model()

    # if df_path.exists():
    #     dataframe = pd.read_pickle(str(df_path))    # Not sure they like Path
    # else:
    times, values = solve(model, dt, interval)

    # dataframe = pd.DataFrame(np.concatenate((times[:,None], values), axis=1))
    # dataframe.to_pickle(str(df_path))

    # times = dataframe.values[:, 0]
    # values = dataframe.values[:, 1:].T
    values = values.T
    plot_ode(times, values[0], "V")

    # sns.set()
    # labels = ("V", "m", "h", "n", "NKo", "NKi", "NNao", "NNai","NClo", "NCli", "vol", "O")
    # labels = range(values.shape[1])
    # for value, label in zip(values, labels):
    #     plot_ode(times, value, label)
