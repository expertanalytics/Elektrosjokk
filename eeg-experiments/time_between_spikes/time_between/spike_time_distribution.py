"""Make a continuous distribution of time between spike times between two electrodes."""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from eegutils import (
    find_peaks,
    get_default_filter,
    read_zhe_eeg,
)

from typing import (
    List,
    Any,
    Tuple,
)

from find_distances import (
    get_positions,
    compute_distances,
)


def get_spike_times(eeg, channel_number, number_of_spikes=3, dt=0.1):
    """Find the peaks, and return the time at which the first `number_of_spikes` occur."""
    peaks, _ = find_peaks(eeg[channel_number], get_default_filter())
    truncated_peaks = peaks[:number_of_spikes]*dt
    assert truncated_peaks.size == number_of_spikes
    return truncated_peaks


if __name__ == "__main__":
    positions = get_positions()
    distance_matrix = compute_distances(positions)        # element [i, j] --> distance i -> j

    all_channels_seizure = read_zhe_eeg(start=4078000, stop=4902000, full=True)

    number_of_spikes = 1000

    pairs = [(3, 2), (10, 4), (47, 18)]
    for ch1, ch2 in pairs:
        first3 = get_spike_times(all_channels_seizure, ch1, number_of_spikes=number_of_spikes)
        second2 = get_spike_times(all_channels_seizure, ch2, number_of_spikes=number_of_spikes)
        distance_between = distance_matrix[ch1, ch2]

        time_distribution = (first3 - second2)/distance_between   # plt.savefig("foo.png")

        # Density Plot and Histogram of all arrival delays
        ax = sns.distplot(
            time_distribution,
            hist=True,
            kde=True,
            color='darkblue',
            hist_kws={'edgecolor': 'black'},
            kde_kws={'linewidth': 4}
        )

        ax.set_xlabel(r"$\Delta$ t [ms]")
        # ax.set_ylabel("frequency")

        # NB! Careful with the wording here
        ax.set_title("Time between spikes at channel {} and {}".format(ch2, ch1))

        fig = ax.get_figure()
        fig.savefig("distribution_plots/distribution_{}to{}.png".format(ch1, ch2))

        plt.close(fig)
