"""Find the time between individual peaks."""

import numpy as np
import matplotlib.pyplot as plt

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

# Get distances, find argsort(distances)
positions = get_positions()
distances = compute_distances(positions)[0]
distance_sort_indices = np.argsort(distances)

# Load the EEG data
all_channels_seizure = read_zhe_eeg(start=4078000, stop=4902000, full=True)
time_interval = 5000*20      # The amount of time to consider. the sampliong frequency is 5kHz
all_channels_seizure = all_channels_seizure[:, :time_interval]
num_time_differences = 100    # The number of time between peak differences to compute

channel_peak_dict = {}      # Store the peaks times as {channel_number: list of peak indices}
for channel_number in range(all_channels_seizure.shape[0]):
    peaks, _ = find_peaks(all_channels_seizure[channel_number], get_default_filter())
    if len(peaks) >= num_time_differences:      # Discard defunct channels. There are some of them.
        channel_peak_dict[channel_number] = peaks


"""
time_difference_list = []
for i in range(num_time_differences):
    difference_list = []
    for channel in channel_peak_dict.keys():
        difference_list.append(
            [channel_peak_dict[channel][i] - channel_peak_dict[j][i] for j in channel_peak_dict.keys()]
        )
    time_difference_list.append(difference_list)
    break
"""


peak_time_distance_dict = {}
for channel_id, peaks in channel_peak_dict.items():
    peak_time_distance_dict[channel_id] = {}
    for peak_number in range(num_time_differences):
        peak_time_distance_dict[channel_id][peak_number] = np.fromiter(
            map(
                lambda x: (peaks[peak_number] - x[peak_number])/5,      # five samples per ms
                channel_peak_dict.values(),
            ),
            dtype="f8"
        )

distance_list = []
time_list = []
for channel_id in (1, 5, 23):
    for peak_number in range(num_time_differences):
        time_difference_array = peak_time_distance_dict[channel_id][peak_number][distance_sort_indices]
        distances_array = distances[distance_sort_indices]

        distance_list.append(distances_array)
        time_list.append(peak_time_distance_dict[channel_id][peak_number][distance_sort_indices])
        # print(time_difference_array / distances_array)
        # print(peak_time_distance_dict[channel_id][peak_number][distance_sort_indices])

distance_array = np.concatenate(distance_list)
time_array = np.concatenate(time_list)
np.save("distance", distance_array)
np.save("time", time_array)
