import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from boxplot_eeg import get_data

import scipy.signal as sig


def plot_data(
        name,
        data,
        title,
        start=0,
        stop=None,
        N=1,
        lines=None,
        ylim=None,
        background=None,
        figsize=None
):
    times = np.linspace(0, data[start:stop:N].size/5000, data[start:stop:N].size)

    if lines is None:
        lines = []

    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(times, data[start:stop:N])

    for l in lines:
        ax.axvline(l/5000, color="k", linestyle="--")

    if ylim is not None:
        ax.set_ylim(ylim)

    if background is not None:
        ax.plot(times, background, alpha=0.5)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\mu V$")
    ax.set_title(title)


    ax.annotate(
        "Stimulus",
        xy=(80, 80),
        xytext=(0.1, 0.90),
        textcoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="top",
        color="r"
    )

    ax.annotate(
        "Seizure",
        xy=(80, 80),
        xytext=(0.5, 0.90),
        textcoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="top",
        color="r"
    )

    ax.annotate(
        "Post Ictal Supression",
        xy=(80, 80),
        xytext=(0.7, 0.90),
        textcoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        color="r"
    )

    fig.savefig("{}.png".format(name))


def make_three_plots():
    data = get_data(full=True)[0, :]
    print(data.shape)
    start = int(4e6)
    plot_data(
        "channel1_stimulus_seizure_supression",
        data,
        "The events of an ECT treatment",
        start=int(4e6),
        N=1,
        lines=[78000, 822000]
    )

    plot_data(
        "channel1_stimulus",
        data,
        "The stimulus",
        start=start,
        stop=start + 78000
    )

    plot_data(
        "channel1_seizure",
        data,
        "The seizure",
        start=start + 78000,
        stop=start + 822000
    )

    plot_data(
        "channel1_supression",
        data,
        "Post ictal supression",
        start=start + 822000
    )


if __name__ == "__main__":
    data = get_data(full=True)
    print(data.shape)

    # start = int(4e6)
    start = int(805*5000)
    stop = start + 8*5000
    plot_data("foo", data[0, :], "stimulus", start=start, N=1, stop=stop)

    # for i in (1, 14, 32, 56):
    #     my_data = data[i, :]
    #     plot_data(
    #         "channel1_stimulus_seizure_supression_{}".format(i),
    #         my_data,
    #         "The events of an ECT treatment",
    #         start=int(4e6),
    #         N=1,
    #         lines=[78000, 822000]
    #     )
