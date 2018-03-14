"""Test the Wei model"""

import math
import matplotlib.pyplot as plt

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


def main():
    """Solve a single cell model on some time frame."""

    model = Wei()
    time = Constant(0.0)
    # params = BasicSingleCellSolver.default_parameters()
    params = SingleCellSolver.default_parameters()

    params["scheme"] = "CN2"
    # solver = BasicSingleCellSolver(model, time, params)
    solver = SingleCellSolver(model, time, params)

    # Assign initial conditions
    vs_, vs = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    # N = 25*(14000 + 3000)

    dt = 0.02
    T = 17000
    N = T/dt + 1

    # N = 6000*0.02
    # dt = 0.02
    interval = (0.0, N)

    start = systime.clock() 
    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for i, ((t0, t1), vs) in enumerate(solutions):        # TODO: Unpacking from 3.6?
        times.append(t1)
        if i % 1000 == 0:
            print(i, vs.vector().norm("l2"))
        values.append(vs.vector().array())
    print("Time to solve: {}".format(start - systime.clock()))

    return times, values


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
    tic = systime.time()
    times, values = main()
    print(systime.time() - tic)
    np.save("solution.npy", values)
    # np.save("initial_condition.npy", values[-1])
    plot_results(times, values, show=False)
