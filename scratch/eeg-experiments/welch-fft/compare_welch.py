from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from scipy.signal import welch
from eegutils import read_zhe_eeg


def get_eeg_value(state="seizure") -> np.ndarray:
    """Read all chanels from the zhi eeg. The hardcoded values correspond to the seizure."""

    index_dict = {
        "seizure": (4078000, 4902000),
        "resting": (0, 5000*200)        # 100 s? sampling frequency is 5000  s^-1
    }

    start, stop = index_dict[state]
    all_channels_seizure = read_zhe_eeg(
        start=start,
        stop=stop,
        full=True
    )
    return all_channels_seizure


def plot_compare_psd(seizure, resting, channel_number):
    nperseg = 1000*5
    seizure_frequencies, seizure_power_density = welch(
        seizure,
        fs=5000,
        window="hann",        # hanning, triang, blackman, hamming, hann,
        nperseg=nperseg,
        noverlap=None,      # None means neperseg/2
        return_onesided=True,
        detrend="constant"
    )
    resting_frequencies, resting_power_density = welch(
        resting,
        fs=5000,
        window="hann",        # hanning, triang, blackman, hamming, hann,
        nperseg=nperseg,
        noverlap=None,      # None means neperseg/2
        return_onesided=True,
        detrend="constant"
    )

    fig, ax = plt.subplots(1)

    ax.plot(seizure_frequencies, seizure_power_density, label="seizure")
    ax.plot(resting_frequencies, resting_power_density, label="resting")

    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequencies [Hz]")
    ax.set_ylabel("Power")
    ax.set_title(f"Psd of EEG channel {channel_number}")
    ax.legend()
    fig.savefig(f"figures/psd_{channel_number}.png")
    plt.close(fig)


if __name__ == "__main__":
    # sns.set()

    seizure = get_eeg_value(state="seizure")
    resting = get_eeg_value(state="resting")
    print("Done loading data")

    defunct_channels = {30, 31, 61}
    for i in range(seizure.shape[0]):
        print(f"Channel {i}")
        if i in defunct_channels:
            continue
        plot_compare_psd(seizure[i, :], resting[i, :], i)
