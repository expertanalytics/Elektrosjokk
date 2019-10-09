import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Custom stuff
from welch import welch_psd
from post import read_point_values
from eegutils import rle

from scipy.signal import find_peaks


def get_probe(*, probe_path: Path, name: str) -> np.ndarray:
    data = read_point_values(probe_path, name)
    return data


def spike_and_burst(voltage_array, times, dt, voltage_threshold=-20):
    """ values grater than `voltage_threshold` is considered a spike. """
    assert len(voltage_array.shape) == 1

    yhat = voltage_array > voltage_threshold         # greater than 30 mV is considered a spike
    runlengths, start_indices, values = rle(yhat)
    spike_indices = runlengths < 10*int(1/dt)       # Factor 1 comes from trial and error
    spike_times = times[start_indices[spike_indices]]       # start times of spikes

    # NB! Need to be careful if it is not continuously spiking
    num_spikes = spike_times.size

    quiet_indices = np.where(runlengths > 2*int(1/dt))[0]     # Why 1000?
    burst_duration = start_indices[quiet_indices - 1]
    return num_spikes, burst_duration[0]


def find_num_spikes(probe_data, time, dt):
    peaks, _ = find_peaks(probe_data, distance=5, height=-55)
    num_peaks = peaks.size
    duration = time[peaks[-1]]
    return num_peaks, duration



def make_report(probe_data):
    time = probe_data[:, 0]
    dt = (time[1] - time[0])/1000

    for i in range(1, probe_data.shape[1]):
        if np.max(probe_data[:, i]) > 10:
            num_spikes = find_num_spikes(probe_data[:, i], time, dt)
        # num_spikes, durations = spike_and_burst(probe_data[:, i], time, dt)
        # if num_spikes != 0:
        #     print(num_spikes, durations)


if __name__ == "__main__":
    datapath = Path("/home/jakobes/data/paper/squiggly/squiggly")

    experiment = "K_2.0_8.0"
    # case = "0eba006d"
    # case = "9769e7c8"
    # probe_path = datapath / experiment / case

    experiment_list = [
        # "K_4.0_10.0",
        "K_2.0_8.0",
        # "K_4.0_8.0",
        # "K_6.0_8.0",
        # "K_4.0_9.5"
    ]

    for experiment in map(Path, experiment_list):
        for case in filter(lambda x: x.is_dir(), (datapath / experiment).iterdir()):
            for probe in filter(lambda x: x.is_dir(), case.iterdir()):
                if "trace0" in probe.stem and not "offset" in probe.stem:
                    probe_path = datapath / experiment / case 
                    probe_name = probe.stem
                    probe_data = get_probe(probe_path=probe_path, name=probe_name)
                    print(case.stem, probe_name)
                    make_report(probe_data)
                    break
