import itertools

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
    get_pairs,
)


def get_spike_times(eeg, channel_number, number_of_spikes=3, dt=0.1):
    """find the peaks, and return the time at which the first `number_of_spikes` occur."""
    peaks, _ = find_peaks(eeg[channel_number], get_default_filter())
    truncated_peaks = peaks[:number_of_spikes]*dt
    if truncated_peaks.size != number_of_spikes:
        print("Channel {}: Expected {} peaks, got {}".format(
            channel_number,
            number_of_spikes,
            truncated_peaks.size
        ))
    # assert truncated_peaks.size == number_of_spikes, "Error in channel {}".format(channel_number)
    return truncated_peaks


if __name__ == "__main__":
    all_channels_seizure = read_zhe_eeg(start=4078000, stop=4902000, full=True)
    number_of_spikes = 100

    position_data = get_positions()
    points = get_pairs(position_data);

    all_time_distance_list = []

    censored_channels = {30, 31, 61}
    for ch1, ch2 in points:
        if len({ch1, ch2} & censored_channels) > 0:
            continue
        first3 = get_spike_times(all_channels_seizure, ch1, number_of_spikes=number_of_spikes)
        second2 = get_spike_times(all_channels_seizure, ch2, number_of_spikes=number_of_spikes)
        distance_between = np.linalg.norm(position_data[ch1] - position_data[ch2])
        if first3.size != second2.size:
            print("Skipping channels ({}, {})".format(ch1, ch2))
            continue

        time_distribution = (first3 - second2)/distance_between

        all_time_distance_list.append(time_distribution)

        ax = sns.distplot(
            time_distribution,
            hist=True,
            kde=True,
            color='darkblue',
            hist_kws={'edgecolor': 'black'},
            kde_kws={'linewidth': 4}
        )

        ax.set_xlabel(r"$\Delta$ t [ms]")
        ax.set_title("Time between spikes at channel {} and {}".format(ch2, ch1))
        fig = ax.get_figure()
        fig.savefig("distribution_plots/distribution_{}to{}.png".format(ch1, ch2))
        plt.close(fig)

    flattened_time_distribution = list(itertools.chain(*all_time_distance_list))
    ax = sns.distplot(
        flattened_time_distribution,
        hist=True,
        kde=True,
        color='darkblue',
        hist_kws={'edgecolor': 'black'},
        kde_kws={'linewidth': 4}
    )

    ax.set_xlabel(r"$\Delta$ t [ms]")
    ax.set_title("Time between spikes at channel {} and {}".format(ch2, ch1))
    fig = ax.get_figure()
    fig.savefig("foo.png".format(ch1, ch2))
    plt.close(fig)
