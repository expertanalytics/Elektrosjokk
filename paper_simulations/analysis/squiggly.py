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
    datapath = Path("/home/jakobes/data/paper/squiggly/squiggly")

    experiment = "K_2.0_8.0"
    # case = "0eba006d"
    # case = "9769e7c8"
    # probe_path = datapath / experiment / case

    experiment_list = [
        # "K_4.0_10.0",
        # "K_2.0_8.0",
        # "K_4.0_8.0",
        # "K_6.0_8.0",
        "K_4.0_9.5"
    ]

    # probe_path = datapath
    # experiment = experiment_list[0]
    # case = "6b74b21a"
    # probe_name = "trace0_1"
    # probe_data = get_probe(probe_path=datapath / experiment / case, name=probe_name)
    # report_probe(probe_data, experiment, case, probe_name)
    # 1/0

    for experiment in map(Path, experiment_list):
        for case in filter(lambda x: x.is_dir(), (datapath / experiment).iterdir()):
            for probe in filter(lambda x: x.is_dir(), case.iterdir()):
                if "trace0" in probe.stem and not "offset" in probe.stem:
                    probe_path = datapath / experiment / case
                    probe_name = probe.stem
                    probe_data = get_probe(probe_path=probe_path, name=probe_name)
                    print("\n", case.stem, probe_name)
                    report_probe(probe_data, experiment, case.stem, probe_name=probe_name)
            input()

    # for theta in [1, 5, 9, 13]:
    #     probe_name = f"trace_offset4_{theta}"
    #     probe_data = get_probe(probe_path=probe_path, name=probe_name)
    #     report_probe(probe_data, experiment, case, probe_name=probe_name)
