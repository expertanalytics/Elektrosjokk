import pickle
import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt


font = {
    "family": "ubuntu",
}

mpl.rc('font', **font)


def plot_data_dict(data, odir="figures"):
    time = np.linspace(0, 100, data.shape[0])

    vol = 1.4368e-15
    voli = data[:, 10]
    beta0 = 7
    volo = (1 + 1/beta0)*vol - voli

    data[:, 4] /= volo
    data[:, 5] /= voli
    data[:, 6] /= volo
    data[:, 7] /= voli
    data[:, 8] /= volo
    data[:, 9] /= voli

    data[:, 10] /= volo

    titles = (
        r"Transmembrane potential", r"Voltage Gate (m)", r"Voltage Gate (n)", r"Voltage Gate (h)",
        r"Extracellular Potassium $[K^+]$", r"Intracellular Potessium $[K^+]$",
        r"Extracellular Sodium $[Na^+]$", r"Intracellular Sodium $[Na^+]$",
        r"Exctracellular Chlorine $[Cl^-]$", r"Intracellular Chlorine $[CL^-]$",
        r"Ratio of intracellular to extracellular volume", r"Extracellular Oxygen $[O_2]$"
        )
    ylabels = ("mV", "mV", "mV", "mV", "mol", "mol", "mol", "mol", "mol", "mol",
               r"$Vol_i/Vol_e$", "mol")
    names = [
        "V",
        "m",
        "h",
        "n",
        "Ko",
        "Ki",
        "Nao",
        "Nai",
        "Clo",
        "Cli",
        "beta",
        "O"
    ]

    for i, (name, _ylabel, title) in enumerate(zip(names, ylabels, titles)):
        # fig, ax1 = plt.subplots(1, figsize=(8, 8))
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(111)
        fig.suptitle(title, size=52)
        # for i in range(3):

        ax1.plot(time, data[:, i], label=f"{name}", linewidth=4)
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


if __name__ == "__main__":
    data = np.load("solution.npy")
    foo = plot_data_dict(data)
