import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from eegutils import (
    find_peaks,
    get_default_filter,
)

from typing import (
    List,
    Any,
    Tuple,
)


def get_data(
        ect_dir_path: str="Documents",
        full: bool=True,
        start: int=0,
        stop: int=None,
        step: int=1,
) -> "np.ndarray":
    """Read the eeg from Zhi. For prototyping, set full to `False`

    Arguments:
        ect_dir_path: The path to `ECT-data` repo.
        full: Load the full pickle (a couple of seconds) or the short npy-file.
        start: return data[:, start:]
        stop: return data[:, :stop]
        step: return data[:, ::step]
    """
    basepath = Path.home() / ect_dir_path / "ECT-data/zhi"
    if full:
        data = pd.read_pickle(basepath / "EEG_signal.xz").values
        print("done reading pickle")
        return data[:, start:stop:step]
    data = np.load(basepath / "SHORT_EEG_signal.npy")
    return data


def make_boxplot(
        data: List[np.ndarray],
        outpath: Path,
        label: List[Any]=None,
        ylim: Tuple[float, float]=None
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data, whis=[5, 95])
    ax.set_xlabel("Channel number", fontsize=20)
    ax.set_ylabel("Time between spikes [ms]", fontsize=20)
    ax.set_title("Variation of time between spikes during a seizure", fontsize=26)

    _label = label
    if label is None:
        _label = list(range(len(data)))
    xtick_names = plt.setp(ax, xticklabels=_label)

    if ylim is not None:
        ax.set_ylim(ylim)

    rotation = 0
    if len(data) > 30:      # Heuristic for space on canvas
        rotation = 90

    plt.setp(xtick_names, rotation=rotation, fontsize=12)      # set property on firt argument
    outpath = Path(outpath)
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig("{}.png".format(outpath))


def boxplot_time_between_spikes():
    start = int(4e6) + 78000
    stop = start + 820000
    data = get_data(full=True, start=start, stop=stop)

    my_filter = get_default_filter()

    time_between_peak_list = []
    box_numbers = [0, 14, 27, 36, 48, 57]
    for channel_number in box_numbers:
        peaks, _ = find_peaks(
            data[channel_number, :],
            my_filter,
            plot=False,
            outpath=Path("channels/channel{}".format(channel_number)),
            sample_frequency=5000
        )
        time_between_peak_list.append(np.diff(peaks)/5)        # scale to [ms]

    sns.set()
    make_boxplot(
        time_between_peak_list,
        label=box_numbers,
        outpath=Path("channels/boxplot"),
        ylim=(0, 300),
    )


if __name__ == "__main__":
    boxplot_time_between_spikes()
