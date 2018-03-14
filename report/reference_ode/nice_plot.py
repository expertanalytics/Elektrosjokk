import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


font = {
    "family": "ubuntu",
}

mpl.rc('font', **font)

TIMESTEP = 1.0


def get_data(filename="wei_solution_dt1s.npy"):
    return np.load(filename)

def plot(data, odir="figures"):
    t = np.linspace(0, data.shape[0] - 1, data.shape[0])/1000   # Time in ms
    t = t[:int(t.size/3)]
    v = data[:, 0]
    m = data[:, 1]
    h = data[:, 2]
    n = data[:, 3]
    NKo = data[:, 4]
    NKi = data[:, 5]
    NNao = data[:, 6]
    NNai = data[:, 7]
    NClo = data[:, 8]
    NCli = data[:, 9]
    voli = data[:, 10]
    O = data[:, 11]

    vol = 1.4368e-15
    beta0 = 7
    volo = (1 + 1/beta0)*vol - voli
    beta = voli/volo

    Ko = NKo/volo
    Ki = NKi/voli
    Nao = NNao/volo
    Nai = NNai/voli
    Clo = NClo/volo
    Cli = NCli/voli

    titles = (
        r"Transmembrane potential", r"Voltage Gate (m)", r"Voltage Gate (n)", r"Voltage Gate (h)",
        r"Extracellular Potassium $[K^+]$", r"Intracellular Potessium $[K^+]$",
        r"Extracellular Sodium $[Na^+]$", r"Intracellular Sodium $[Na^+]$",
        r"Exctracellular Chlorine $[Cl^-]$", r"Intracellular Chlorine $[CL^-]$",
        r"Ratio of intracellular to extracellular volume", r"Extracellular Oxygen $[O_2]$"
        )
    ylabels = ("mV", "mV", "mV", "mV", "mol", "mol", "mol", "mol", "mol", "mol",
               r"$Vol_i/Vol_e$", "mol")
    names = ("V", "m", "h", "n", "Ko", "Ki", "Nao", "Nai", "Clo", "Cli", "beta", "O")
    data_tuple = (v, m, h, n, Ko, Ki, Nao, Nai, Clo, Cli, beta, O)
    for name, _ylabel, title, data in zip(names, ylabels, titles, data_tuple):
        # fig, ax1 = plt.subplots(1, figsize=(8, 8), dpi=100)
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(111)
        fig.suptitle(title, size=52)

        ax1.plot(t, data[:t.size], label=f"{name}", linewidth=4)
        ax1.set_ylabel(_ylabel, size=48)
        ax1.set_xlabel("Time (s)", size=48)
        ax1.grid()

        x0,x1 = ax1.get_xlim()
        y0,y1 = ax1.get_ylim()
        ax1.set_aspect((x1-x0)/(y1-y0))

        tx = ax1.xaxis.get_offset_text()
        tx.set_fontsize(28)
        ty = ax1.yaxis.get_offset_text()
        ty.set_fontsize(28)

        ax1.tick_params(axis="both", which="major", labelsize=28)
        ax1.tick_params(axis="both", which="minor", labelsize=28)
        # ax1.legend(loc="best", prop={"size": 24})
        fig.savefig(f"{odir}/{name}_ode.png")
        plt.close(fig)


if __name__ == "__main__":
    data = get_data("wei_solution_dt1.npy")
    plot(data)
