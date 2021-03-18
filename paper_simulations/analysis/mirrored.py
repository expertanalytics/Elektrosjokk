import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from pathlib import Path

# Custom stuff
from welch import welch_psd
from post import read_point_values
from eegutils import rle


def get_probe(*, probe_path: Path, name: str) -> np.ndarray:
    data = read_point_values(probe_path, name)
    return data


def find_num_spikes(probe_data, time, dt):
    # peaks, _ = find_peaks(probe_data, distance=5, height=-55, prominence=100)
    peaks, properties = find_peaks(probe_data, height=-55, prominence=20)
    num_peaks = peaks.size
    if num_peaks == 0:
        return num_peaks, 0
    prominences = properties["prominences"]
    duration = time[peaks[-1]]
    return num_peaks, duration


def report_probe(probe_data, experiment, case, probe_name):
    outdir = Path("figures") / experiment / case / probe_name
    outdir.mkdir(exist_ok=True, parents=True)

    time = probe_data[:, 0]
    time_indices = time <= 10000
    time = time[time_indices]
    dt = time[1] - time[0]
    # for i in range(1, probe_data.shape[1]):
    for i in (24, 27):
        data = probe_data[time_indices, i]
        num_spikes, durations = find_num_spikes(data, time, dt)
        print(f"num spikes: {num_spikes}, duration: {durations}")

        fig, ax = plt.subplots(1)
        ax.plot(time, data)
        fig.suptitle(f"{probe_name} -- spikes: {num_spikes}, duration: {durations}")
        fig.savefig(outdir / f"probe{i}.png")
        plt.close(fig)


if __name__ == "__main__":
    datapath = "/home/jakobes/data/mirrored/bf9fb5fb"
