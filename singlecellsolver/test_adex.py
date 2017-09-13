import math
import matplotlib.pyplot as plt
from xalbrain import *
import time as systime
import seaborn as sns

sns.set()

# For easier visualization of the variables
parameters["reorder_dofs_serial"] = False

# For computing faster
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags


class Stimulus(Expression):
    "Some self-defined stimulus."
    def __init__(self, **kwargs):
        self.t = kwargs["t"]
    def eval(self, value, x):
        if float(self.t) >= 0 and float(self.t) <= 2:
            value[0] = -1*(-100)
        else:
            value[0] = 0.0


def main():
    "Solve a single cell model on some time frame."

    # Initialize model and assign stimulus
    # model = Adex()
    model = AdexManual()
    time = Constant(0.0)
    model.stimulus = Stimulus(t=time, degree=1)

    # Init solver
    adex_solver = True 
    params = SingleCellSolver.default_parameters()
    if adex_solver:
        params = SingleCellSolver.default_parameters_adex()

    # params["theta"] = 0.5     # FIXME: Why not a parameter?
    params["scheme"] = "RK4"
    solver = SingleCellSolver(model, time, params, adex_solver)
    
    # Assign initial conditions
    (vs_, vs) = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    dt = 0.005
    interval = (0.0, 10.0)

    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for ((t0, t1), vs) in solutions:
        times.append(t1)
        values.append(vs.vector().array())

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

    """
    rows = int(math.ceil(math.sqrt(len(variables))))
    for (i, var) in enumerate(variables):
        plt.subplot(rows, rows, i+1)
        plt.plot(times, var, '*-')
        plt.title("Var. %d" % i)
        plt.xlabel("t")
        plt.grid(True)
        outfile.write(str(var))
        outfile.write('\n')
    info_green("Saving plot to 'variables2.pdf'")
    outfile.close()
    plt.savefig("variables2.pdf")
    if show:
        plt.show()
    """

if __name__ == "__main__":
    tic= systime.time()
    times, values = main()
    print(systime.time() - tic)
    plot_results(times, values, show=False)
