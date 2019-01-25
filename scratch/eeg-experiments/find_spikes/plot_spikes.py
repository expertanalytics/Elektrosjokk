"""Find and plot the spikes in a short time interval."""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from eegutils import (
    find_peaks,
    get_default_filter,
    read_zhe_eeg,
)


data = read_zhe_eeg(full=False)
ch0 = data[0, 200000:200000 + 2*5000]

cutoff_frequency = 60
lowpass_filter = get_default_filter(cutoff_frequency=cutoff_frequency)
figpath = Path("figures/peaks{}".format(cutoff_frequency))
peaks, _ = find_peaks(ch0, lowpass_filter, sample_frequency=5000, plot=True, outpath=figpath)






# # plt.plot(ch0)
# # plt.savefig("foo.png")


# print(peaks)

# time = np.linspace(0, ch0.size/5000, ch0.size)        # in ms

# fig, ax = plt.subplots(1)

# low_pass_ch0 = lowpass_filter(ch0)

#         ax.plot(time, data, label="raw eeg")
#         ax.plot(
#             time,
#             low_pass_data,
#             alpha=0.75,
#             linestyle="-.",
#             label="smooth"
#         )

#         # Plotting red dots for peaks
#         ax.plot(time[peaks], data[peaks], "o", color="r", alpha=0.75)

#         # Set labels and such
#         ax.set_title("Peaks in the EEG signal", fontsize=20)
#         ax.set_ylabel(r"$\mu V$")
#         ax.set_xlabel("Time [s]")
#         ax.yaxis.grid(True)
#         ax.legend(fontsize=12)
