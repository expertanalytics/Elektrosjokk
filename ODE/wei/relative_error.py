import numpy as np


def relative_error(a, b):
    """Treat `b` as the reference."""
    return numpy.linalg.norm(a - b)/np.linalg.norm(b)



if __name__ == "__main__":
    import pandas as pd
    fine = pd.read_pickle("ODE_SOLUTION.xz")
    coarse = pd.read_pickle("ode_solution.xz")

    err = relative_error(coarse, fine)
    print(err)
