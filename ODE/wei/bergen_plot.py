import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12


def plot_V(data):
    """Plot transmembrane potential."""
    N = 60000*100
    time = data[N:, 0]/1000
    V = data[N:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, V)
    ax.set_title("Transmembrane Potential", fontsize=26)
    ax.set_xlabel("Time [s]", fontsize=20)
    ax.set_ylabel("V [mv]", fontsize=20)

    fig.savefig("nice_plots/V.png")


def plot_long_V(data):
    """Plot transmembrane potential."""
    # N = 60000*100
    time = data[:, 0]/1000
    V = data[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, V)
    ax.set_title("Transmembrane Potential", fontsize=26)
    ax.set_xlabel("Time [s]", fontsize=20)
    ax.set_ylabel("V [mv]", fontsize=20)

    fig.savefig("nice_plots/V_long.png")


def plot_Ions(data):
    """Plot extracellular Potsssium, Sodium and Cloride."""
    N = 60000*100
    time = data[N:, 0]/1000     # To seconds
    vol = data[N:, 11]   # vol

    NKo = data[N:, 5]     # Ko
    NNao = data[N:, 7]     # NNao
    NClo = data[N:, 9]     # NClo

    fig = plt.figure()
    ax = fig.add_subplot(111)

    Ko = NKo/vol
    Nao = NNao/vol
    Clo = NClo/vol

    ax.plot(time, Ko, label="K")
    ax.plot(time, Nao, label="Na")
    ax.plot(time, Clo, label="Cl")

    ax.set_title("Extracellular $K^+$, $Na^+$ and $Cl^-$", fontsize=26)
    ax.set_xlabel("Time [s]", fontsize=20)
    ax.set_ylabel("mM", fontsize=20)

    ymin = min(np.min(Ko), np.min(Nao), np.min(Clo))
    ymax = max(np.max(Ko), np.max(Nao), np.max(Clo))

    ax.vlines(time[632488], ymin, ymax, colors="k", linestyles="--")
    ax.vlines(time[1218868], ymin, ymax, colors="k", linestyles="--")

    ax.legend(fontsize=12)

    fig.savefig("nice_plots/ions.png")


if __name__ == "__main__":
    data = np.load("ode_values.npy")
    # plot_long_V(data)
    # plot_V(data)
    plot_Ions(data)
