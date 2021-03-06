import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.signal import find_peaks

# Custom stuff
from welch import welch_psd
from post import read_point_values
from eegutils import rle
from powerlaw import PowerLawObjective


def get_probe(*, probe_path: Path, name: str):
    data = read_point_values(probe_path, name)
    return data


def get_psd(*, data: np.ndarray, FS, NPERSEG):
    # dt should be in seconds
    if len(data.shape) == 1:
        signal = data
    else:
        signal = data.sum(axis=1) / data.shape[1]

    frequencies, power_density = welch_psd( eeg_signal=signal,
        fs=FS,
        nperseg=NPERSEG,
        log_scale=False
    )
    return frequencies, power_density


def find_num_spikes(probe_data, time, dt):
    peaks, properties = find_peaks(probe_data, height=-55)

    num_peaks = peaks.size
    if num_peaks == 0:
        return num_peaks, 0
    duration = time[peaks[-1]]
    return num_peaks, duration


def plot_psd(datapath, experiment, simulation_hash):
    probe_path = datapath / experiment / simulation_hash

    probe_name = f"trace_sub0"
    probe_data = get_probe(probe_path=probe_path, name=probe_name)
    if probe_data is None:
        print("cannot find data")
    time = probe_data[:, 0]
    dt = (time[1] - time[0])/1000


    for i in range(1, probe_data.shape[1]):
        frequencies, power_density = get_psd(data=probe_data[:, i], FS=1/dt, NPERSEG=500)
        fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)
        fig.suptitle(f"{probe_name} -- i: {i}")

        ax1.plot(time, probe_data[:, i], linewidth=0.2)
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("mV")
        ax1.set_ylim([-80, 60])
        ax1.plot(time, np.ones_like(time)*-40, color="r", linestyle="--", linewidth=0.4)
        ax2.plot(frequencies, power_density)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Hz")
        ax2.set_ylabel("dB")

        outpath = Path("figures") / experiment / simulation_hash
        outpath.mkdir(exist_ok=True, parents=True)
        fig.savefig(outpath / Path(f"{probe_name}_i{i}.png"))
        plt.close(fig)


def make_report(datapath, experiment, simulation_hash):
    probe_path = datapath / experiment / simulation_hash

    probe_name = f"trace_sub0"
    probe_data = get_probe(probe_path=probe_path, name=probe_name)
    if probe_data is None:
        print("cannot find data")
    time = probe_data[:, 0]
    dt = (time[1] - time[0])/1000

    for i in range(1, probe_data.shape[1]):
        num_spikes, durations = find_num_spikes(probe_data[:, i], time, dt)
        print(num_spikes, durations)


if __name__ == "__main__":

    datapath = Path("/home/jakobes/data/paper/concentric_circle")
    experiment = "4.0_8.0"

    for e in (datapath / experiment).iterdir():
        if e.is_dir():
            print(e.stem,)
            make_report(datapath, experiment, e.stem)
            # plot_psd(datapath, experiment, e.stem)
            print()
            print()

    # simulation_hash = "041398b8"
    # plot_psd(datapath, experiment, simulation_hash)
    # make_report(datapath, experiment, simulation_hash)
