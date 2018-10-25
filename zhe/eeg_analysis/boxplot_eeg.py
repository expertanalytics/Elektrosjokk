import time

import numpy as np
import pandas as pd

from pathlib import Path

from itertools import groupby


def get_data(ect_dir_path = "Documents", full = False):
    if full:
        datapath = Path.home() / ect_dir_path / "ECT-data/zhi/EEG_signal.xz"
        data = pd.read_pickle(datapath).values
        print("done reading pickle")
        return data
    datapath = Path.home() / ect_dir_path / "ECT-data/zhi/SHORT_EEG_signal.npy"
    return np.load(datapath)


def runs_of_ones(bits):
    for bit, group in groupby(bits):
        if bit:
            yield sum(group)


def runs_of_ones_array(bits):
    bounded = np.hstack(([0], bits, [0]))
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts


def rle(inarray):
    """ returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)
    n = ia.size
    y = np.array(ia[1:] != ia[:-1])
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return z, p, ia[i]


def count_adjecent_true(arr):
    assert len(arr.shape) == 1
    assert arr.dtype == np.bool
    if arr.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    sw = np.insert(arr[1:] ^ arr[:-1], [0, arr.shape[0] - 1], values=True)
    sw1 = np.arange(sw.shape[0])[sw]
    offset = 0 if arr[0] else 1
    lengths = sw1[offset + 1::2] - sw1[offset:-1:2]
    return sw1[offset:-1:2], lengths



def test_all():
    test_array = np.array(
        [
            1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1
        ],
        dtype=np.bool
    )

    correct = np.array([1, 2, 3, 1, 8, 2, 3])

    print(correct)
    print(np.fromiter(runs_of_ones(test_array), dtype="i8"))
    z, p, a = rle(test_array)
    print(runs_of_ones_array(test_array))
    print(z[a])
    print(count_adjecent_true(test_array)[1])


def timer(f, arg):
    tick = time.clock()
    f(arg)
    tock = time.clock()
    return tock - tick


def test_timing(data):

    def rle_wrapper(data):
        z, p, a = rle(data)
        return z[a]

    time_dict = {
        "runs_of_ones": timer(lambda x: np.fromiter(runs_of_ones(x), dtype="i8"), data),
        "runs_of_ones_array": timer(runs_of_ones_array, data),
        "rle": timer(lambda x: rle_wrapper(x), data),
        "count_adjecent_true": timer(lambda x: count_adjecent_true(x)[1], data)
    }
    for key in sorted(time_dict, key=time_dict.get):
        print(key, time_dict[key])


def make_boxplot(data):
    pos_list = []
    for channel in data:
        channel_max_indices = channel == channel.max()
        runlengths, startpositions, values = rle(channel_max_indices)
        positive_run_length = runlengths[values]
        pos_list.append(positive_run_length)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(np.asarray(pos_list))
    ax.set_xlabel("channel number", fontsize=20)
    ax.set_ylabel("time [ms]", fontsize=20)


    xtick_names = plt.setp(ax, xticklabels=np.arange(data.shape[0]))
    plt.setp(xtick_names, rotation=90, fontsize=8)      # set property on firt argument
    fig.savefig("boxplot_eeg.png")
    ax.set_ylim([-50, 400])
    fig.savefig("boxplot_eeg_ylim.png")



if __name__ == "__main__":
    data = get_data(full=True)
    data /= 5       # Convert to ms

    # test_all()
    # test_timing(data[0, :])
    make_boxplot(data)
