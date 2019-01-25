from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


if __name__ == "__main__":
    start = 4078000 + 5000
    data = read_zhe_eeg(
        start=start,
        stop=start + 2*15000,
        full=True
    )
    
    print(data.shape)
    y1 = data[0, :]
    y2 = data[1, :]
    diff = np.abs(y1 - y2)

    time = np.linspace(0, 6, diff.size)
    # plt.plot(time, y1, time, y2)
    plt.plot(time, diff)
    plt.savefig("comparison.png")
