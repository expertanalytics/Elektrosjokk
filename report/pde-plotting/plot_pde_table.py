import pickle
import numpy as np
import matplotlib as mpl

mpl.use("agg")

import matplotlib.pyplot as plt


font = {
    "family": "ubuntu",
}

mpl.rc('font', **font)


def unpickle_data(filename="time_samples.pickle"):
    with open(filename, "rb") as in_handle:
        data_dict = pickle.load(in_handle)
    return data_dict


def plot_data_dict(data_dict, odir="figures"):
    points = data_dict["points"]
    time = tuple(filter(lambda x: x != "points", data_dict.keys()))

    function_dict = {key: [] for key in data_dict[time[0]].keys()}
    for t in time:
        for key in data_dict[t]:
            function_dict[key].append(np.array(data_dict[t][key]))
    np.vstack(function_dict["vs-0"])
    plot_dict = {key: np.vstack(value) for key, value in function_dict.items()}

    vol = 1.4368e-15
    voli = plot_dict["vs-10"]
    beta0 = 7
    volo = (1 + 1/beta0)*vol - voli
    plot_dict["vs-10"] = voli/volo

    plot_dict["vs-4"] /= volo
    plot_dict["vs-5"] /= voli
    plot_dict["vs-6"] /= volo
    plot_dict["vs-7"] /= voli
    plot_dict["vs-8"] /= volo
    plot_dict["vs-9"] /= voli

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

    ylimits = (     # This did not wok particularity well
        (-140, -40),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 12),
        (100, 150),
        (140, 160),
        (15, 20),
        (110, 135),
        (5, 10),
        (6, 8),
        (29, 30)
    )

    t = np.array(time)/1000     # Convert from (ms) to (s)
    for name, _ylabel, title, ylim in zip(names, ylabels, titles, ylimits):
        # fig, ax1 = plt.subplots(1, figsize=(8, 8))
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(111)
        fig.suptitle(title, size=52)
        # for i in range(3):
        i = 0
        data = plot_dict[names[name]][:, i]        # axis 1 is over sample points

        ax1.plot(t, data, label=f"{name}", linewidth=4)
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
    my_data = unpickle_data()
    foo = plot_data_dict(my_data)