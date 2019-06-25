import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

from pathlib import Path

from post import (
    Loader,
    read_point_metadata,
    read_point_values,
    load_times
)

from postspec import (
    PlotSpec,
    LoaderSpec,
)

from postplot import plot_point_field

from eegutils import rle


def make_report(casedir, dim, verbose=False):
    loader_spec = LoaderSpec(casedir)
    loader = Loader(loader_spec)

    times = load_times(casedir)
    point_name = "point_v"
    point_metadata = read_point_metadata(casedir, point_name)
    point_values = read_point_values(casedir, point_name)

    def rle_stuff(input_array, times, dt=0.1):
        assert len(input_array.shape) == 1

        yhat = input_array > 30        # greater than 30 mV is considered a spike
        runlengths, start_indices, values = rle(yhat)
        spike_indices = runlengths < 1*int(1/dt)       # Factor 10 comes from inspection of data
        spike_times = times[start_indices[spike_indices]]       # start times of spikes

        # NB! Need to be careful if it is not continuously spiking
        num_spikes = spike_times.size

        quiet_indices = np.where(runlengths > 1000*int(1/dt))[0]     # Why 100?
        burst_start_times = times[start_indices[quiet_indices[:-1] + 1]]
        burst_stop_times = times[start_indices[quiet_indices[1:]]]
        burst_durations = burst_stop_times - burst_start_times

        return num_spikes, burst_durations

    with open(casedir / "report.md", "w") as of_handle:
        for i in range(point_values.shape[1]):
            if verbose:
                print("rle {}".format(i))
            num_spikes, burst_durations = rle_stuff(point_values[:, i], times.time)
            num_bursts = len(burst_durations)
            line = "probe {}: num_spikes: {}, num_bursts: {}, burst_durations: {}\n"
            of_handle.write(line.format(i, num_spikes, num_bursts, burst_durations))

    def make_domain_plot(my_path, time, y, counter, dim):
        if dim == 1:
            fig, ax = plt.subplots(1)
            x = np.linspace(0, 1, y.size)
            ax.plot(x, y)
            ax.set_title("T: {}".format(time))
            ax.set_ylim(-80, 40)
            fig.savefig(my_path / "field_v{}.png".format(counter))
            plt.close(fig)
        elif dim == 2:
            # TODO: Fix this
            pass
            # plot()

    figpath = casedir / "figures"
    domainpath = figpath / "domain"
    probepath = figpath / "probe"
    domainpath.mkdir(exist_ok=True, parents=True)
    probepath.mkdir(exist_ok=True, parents=True)

    try:
        for i, (t, v) in enumerate(loader.load_field("v")):
            if verbose:
                print("domain {}".format(i))
            make_domain_plot(domainpath, t, v, i, dim=dim)
    except Exception as e:
        print(e)
        print("Skipping domain plots.")

    for i in range(point_values.shape[1]):
        try:
            if verbose:
                print("probe {}".format(i))
            plot_spec = PlotSpec(
                outdir=probepath,
                name="pointV{}".format(i),
                title="pointV",
                ylabel="mV"
            )
            y = point_values[:, i]
            time_array = np.linspace(0, times.time[-1], y.size)
            plot_point_field(times=time_array, probes=(point_values[:, i],), spec=plot_spec)
        except Exception as e:
            print(e)
            print("Skipping point probe")


if __name__ == "__main__":
    my_case_dir = Path("test_dt_N_outdir")
    for p in my_case_dir.iterdir():
        make_report(p, dim=1, verbose=True)