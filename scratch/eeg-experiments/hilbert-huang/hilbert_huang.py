from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch
from eegutils import read_zhe_eeg

from scipy.signal import hilbert

from pyhht import EMD
from pyhht.visualization import plot_imfs

def get_eeg_value(state="seizure") -> np.ndarray:
    """Read all chanels from the zhi eeg. The hardcoded values correspond to the seizure."""

    index_dict = {
        "seizure": (4078000, 4835000),
        "resting": (0, 5000*200)        # 100 s? sampling frequency is 5000  s^-1
    }

    start, stop = index_dict[state]
    all_channels_seizure = read_zhe_eeg(
        start=start,
        stop=stop,
        full=True
    )
    return all_channels_seizure


def compute_hh(data):
    # modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    # x = modes + t
    decomposer = EMD(data)
    imfs = decomposer.decompose()

    t = np.linspace(0, 1, data.size)
    foo = plot_imfs(data, imfs, t) #doctest: +SKIP

    from IPython import embed; embed()

    # plt.savefig("hilbert.png")



if __name__ == "__main__":

    eeg_data = get_eeg_value(state="seizure")
    compute_hh(eeg_data[0, :4*5000])
