import numpy as np

from xalbrain.cellmodels import Wei

from ect.plotting import plot_cell_model

from ect.utilities import preprocess_wei

from ect.specs import Plot_spec


def load_txt(filename="filename"):
    return np.loadtxt(filename, delimiter=",")


if __name__ == "__main__":
    params = Wei.default_parameters()
    pde_data = load_txt("time_samples.txt")
    ode_data = np.load("solution.npy")

    time = np.arange(pde_data.shape[0])

    pde_processed = list(preprocess_wei(pde_data, params))
    ode_processed = list(preprocess_wei(ode_data, params))

    for pdata, odata in zip(pde_processed, ode_processed):
        line = [(pdata.data, "pde"), (odata.data, "ode")]
        plot_cell_model(
            Plot_spec(line, pdata.label, pdata.title, pdata.ylabel),
            time,
            "figures"
        )
