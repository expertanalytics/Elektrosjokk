import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pathlib import Path

from itertools import islice

from typing import (
    Tuple,
    List,
    Iterable,
)

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

from postplot import (
    plot_point_field,
    mplot_function,
)

from postutils import (
    circle_points,
)

from eegutils import (
    rle,
)


def indicators(
    *,
    input_array: np.ndarray,
    times: np.ndarray,
    sampling_interval: float
) -> Tuple[int, List[float]]:
    """Compute the number of spikes and durations of bursts in the signal `input_array`.

    A good value for the sampling interval is 0.1 ms
    """
    assert len(input_array.shape) == 1

    yhat = input_array > 30        # greater than 30 mV is considered a spike
    runlengths, start_indices, values = rle(yhat)
    spike_indices = runlengths < 1*int(1/sampling_interval)       # Factor 10 comes from inspection of data
    spike_times = times[start_indices[spike_indices]]       # start times of spikes

    # NB! Need to be careful if it is not continuously spiking
    num_spikes = spike_times.size

    quiet_indices = np.where(runlengths > 1000*int(1/sampling_interval))[0]     # Why 100?
    burst_start_times = times[start_indices[quiet_indices[:-1] + 1]]
    burst_stop_times = times[start_indices[quiet_indices[1:]]]
    burst_durations = burst_stop_times - burst_start_times

    return num_spikes, burst_durations


def table(
    *,
    loader: Loader,
    point_name: str,
    sampling_interval: float,
    start: int = 0,
    stop: int = None,
    step: int = 1,
):
    times = load_times(loader.casedir)
    point_metadata = read_point_metadata(loader.casedir, point_name)
    point_values = read_point_values(loader.casedir, point_name)

    if stop is None:
        stop = point_values.shape[1]

    with (loader.casedir / "report" / "report.md").open("w") as of_handle:
        for i in range(start, stop, step):
            num_spikes, burst_durations = indicators(
                input_array=point_values[:, i],
                times=times.time,
                sampling_interval=sampling_interval
            )
            num_bursts = len(burst_durations)
            line = "probe {}: num_spikes: {}, num_bursts: {}, burst_durations: {}\n"
            of_handle.write(line.format(i, num_spikes, num_bursts, burst_durations))


def space_plot(
    *,
    loader: Loader,
    field_name: str,
    vextrema: Tuple[float, float] = (-80, 40),
    start: int = 0,
    stop: int = None,
    step: int = 1,
) -> None:
    """Plot the solution in space at a time point."""
    vmax, vmin = vextrema
    figure_path = loader.casedir / "report" / "figures" / "domain"
    figure_path.mkdir(exist_ok=True, parents=True)

    field_iterator = enumerate(islice(loader.load_field(field_name), start, stop, step))
    for i, (time, solution) in field_iterator:
        fig, ax = mplot_function(solution, vmax, vmin, colourbar_label="Transmembrane voltage")
        ax.set_title("T: {}".format(time))
        fig.savefig(figure_path / "domain{}.png".format(i))
        plt.close(fig)


def point_plot(*, loader: Loader, field_name: str, start: int = 0, stop: int = None, step: int = 1) -> None:
    """Plot the solution over time at a discrete set of predefined points."""
    times = load_times(loader.casedir)

    # Metadata contains the spatial coordinates of the points.
    point_metadata = read_point_metadata(loader.casedir, field_name)
    point_values = read_point_values(loader.casedir, field_name)

    if stop is None:
        stop = point_values.shape[1]

    for i in range(start, stop, step):
        plot_spec = PlotSpec(
            outdir=loader.casedir / "report" / "figures" / "points",
            name="pointV{}".format(i),      # Not sure I like the name here
            title="Point V",
            ylabel="mV"
        )
        time_array = np.linspace(0, times.time[-1], point_values.shape[0])
        plot_point_field(times=time_array, probes=(point_values[:, i],), spec=plot_spec)


def psd(*, loader: Loader, field_name: str, sampling_frequency: int):
    """Load several points, and someow compute psd.

    samppling frequency is samples per second, not milli second.
    """
    times = load_times(loader.casedir)

    # Metadata contains the spatial coordinates of the points.
    point_metadata = read_point_metadata(loader.casedir, field_name)
    point_values = read_point_values(loader.casedir, field_name)

    # average_signal = point_values.sum(axis=1) / point_values.shape[1]
    average_signal = point_values.sum(axis=1)

    from welch import welch_eeg, plot_psd
    psd_fig = plot_psd(
        eeg_signal=average_signal,
        fs=sampling_frequency,
        # ylim=(0, 1e3),
        signal_identifier=field_name
    )
    figure_path = Path(loader.casedir) / "report" / "figures" / "psd"
    figure_path.mkdir(exist_ok=True, parents=True)
    psd_fig.savefig(figure_path / "{name}_psd.png".format(name=field_name))




if __name__ == "__main__":
    casedir = Path("/home/jakobes/.simulations/cressman2DCSF") / "d902c7e2"

    loader_spec = LoaderSpec(casedir)
    loader = Loader(loader_spec)

    # space_plot(loader=loader, field_name="v", stop=None)
    # point_plot(loader=loader, field_name="point_v", stop=None)
    # table(loader=loader, point_name="point_v", sampling_interval=0.025, stop=10)
    psd(loader=loader, field_name="psd_v", sampling_frequency=20000)
    # TODO: Add a filter on which time steps to plot. Some kind of range structure.
