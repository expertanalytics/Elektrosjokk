import dolfin as df
import numpy as np
import pandas as pd
import xalbrain as xb
import seaborn as sns
import matplotlib.pyplot as plt

from post import Saver

from postspec import (
    FieldSpec,
    PostProcessorSpec,
)

from postfields import (
    Field,
    PointField,
)

from xalbrain.cellmodels import Cressman

from typing import (
    Tuple,
)


def plot_ode(time, value, label, dt, N=10):
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("ODE plot", size=52)

    ax = fig.add_subplot(111)

    ax.plot(time[::N]*dt, value[::N], label=label, linewidth=2)
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


def get_post_processor(
        outpath: str,
        time_stamp: bool=True,
        home: bool=False
) -> Saver:
    _outpath = Path(outpath)
    if time_stamp:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        _outpath /= timestr
    if home:
        _outpath = Path.home() / _outpath

    pp_spec = PostProcessorSpec(casedir=_outpath)
    saver = Saver(pp_spec)

    field_spec = FieldSpec(save_as=("hdf5", "xdmf"), stride_timestep=1)   # every ms
    points = np.zeros(1)

    saver.add_field(PointField("v", field_spec, points))
    saver.add_field(PointField("m", field_spec, points))
    saver.add_field(PointField("h", field_spec, points))
    saver.add_field(PointField("n", field_spec, points))
    saver.add_field(PointField("Ca", field_spec, points))
    saver.add_field(PointField("K", field_spec, points))
    saver.add_field(PointField("Na", field_spec, points))
    return saver


def solve(model: xb.CardiacCellModel, dt: float, interval: Tuple[float, float]) -> None:
    time = df.Constant(0)

    # Set up solver
    solver_params = xb.SingleCellSolver.default_parameters()
    solver_params["point_integral_solver"]["newton_solver"]["relative_tolerance"] = 1e-13
    solver_params["scheme"] = "GRL1"
    solver = xb.SingleCellSolver(model, time, params=solver_params)

    # Set custom initial values
    reference_ic = np.load("reference_scipy_solution.npy")[:, -1]
    field_names = ("V", "m", "h", "n", "Ca", "K", "Na")
    ic_dict = {n: v for n, v in zip(field_names, reference_ic)}
    model.set_initial_conditions(**ic_dict)

    # Assign initial conditions
    vs_, vs = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    saver = get_post_processor(outpath="out_cressman", time_stamp=True)

    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for i, ((t0, t1), vs) in enumerate(solutions):
        print("Current time: {:5.3e} / {:2.2e}".format(t1, interval[-1]))
        v, m, h, n, Ca, K, Na = vs.split(deepcopy=True)
        update_dict = {"v": v, "m": m, "h": h, "n": n, "Ca": Ca, "K": K, "Na": Na}
        saver.update(time, i, update_dict)
        _values = vs.vector().array()
        assert not np.isnan(_values).any()
        # times.append(t1)
        # print(_values[:_values.size//2])
        values.append(_values)
    saver.close()
    return np.asarray(times), np.asarray(values)


if __name__ == "__main__":
    df.set_log_level(100)
    import time

    from pathlib import Path
    df_path = Path("cressman_solution.xz")

    dt = 5e-2
    T = 7e3
    # T = 10*dt
    interval = (0.0, T)
    model = get_model()

    # if df_path.exists():
    #     dataframe = pd.read_pickle(str(df_path))    # Not sure they like Path
    # else:
    tick = time.clock()
    times, values = solve(model, dt, interval)
    tock = time.clock()
    print("Time: ", tock - tick)

    # dataframe = pd.DataFrame(np.concatenate((times[:,None], values), axis=1))
    # dataframe.to_pickle(str(df_path))

    # times = dataframe.values[:, 0]
    # values = dataframe.values[:, 1:].T
    # values = values.T
    # plot_ode(times, values[0], "V", dt)

    # sns.set()
    # labels = ("V", "m", "h", "n", "NKo", "NKi", "NNao", "NNai","NClo", "NCli", "vol", "O")
    # labels = range(values.shape[1])
    #     plot_ode(times, value, label)
