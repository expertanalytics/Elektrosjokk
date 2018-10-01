"""Plot point fields."""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import count

import yaml


def argmax(iterable):
    return max(zip(iterable, count()))[1]


def get_latest_casedir(directory="out_wei"):
    basedir = Path(directory)

    casedir_list = list(basedir.iterdir())
    numeric_casedir = (int(x.name.replace("-", "")) for x in casedir_list)
    return casedir_list[argmax(numeric_casedir)]


def get_data(name, basedir="out_wei"):
    basepath = get_latest_casedir(basedir)
    basepath /= "point_{}/probes_point_{}.npy".format(name, name)
    return np.load(basepath)


def get_metadata(name, basedir="out_wei"):
    basepath = get_latest_casedir(basedir)
    basepath /= "point_{}/metadata_point_{}.yaml".format(name, name)

    with open(basepath, "r") as in_handle:
        return yaml.load(in_handle)


def get_times(basedir="out_wei"):
    basepath = get_latest_casedir(basedir)
    basepath /= "times.npy"
    return np.load(basepath)


def plot_probes(name, times, probe_data, points, stride=1, basedir="out_wei"):
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    if isinstance(points, tuple):
        points = [points]

    for i, p in enumerate(points):
        ax.plot(
            times[::stride]/1e3,
            probe_data[:, i],
            label="{}".format(p[0]),
            linewidth=2
        )
    ax.legend(fontsize=16)
    ax.set_title("{}".format(name), fontsize=30)
    ax.set_xlabel("Time [s]", fontsize=22)
    ax.set_ylabel("Potential [mV]", fontsize=22)

    outdir = get_latest_casedir(basedir)
    fig.savefig(str(outdir / "{}.png".format(name)))


if __name__ == "__main__":
    import seaborn as sns
    sns.set()
    name_list = ["NClo", "NKo", "NNao", "O", "u", "v", "Vol"]

    times = get_times()
    for name in name_list:
        metadata = get_metadata(name)
        points = metadata["point"]
        stride = metadata["stride_timestep"]
        probe_data = get_data(name)
        plot_probes(name, times, probe_data, points[::2], stride=stride)
