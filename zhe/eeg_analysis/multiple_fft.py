import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from analyse_eeg import get_data

from eegutils import get_default_filter


def multiple_fft(data, window=5000):
    _data = data[:(data.size//window)*window]
    _data.shape = (-1, window)
    avg_data = _data.sum(axis=0)
    avg_data /= data.shape[0]

    sp = np.fft.rfft(avg_data).real[:window//2]
    squared_fft = np.power(sp, 2)
    t = np.fft.rfftfreq(avg_data.size, d=1/window)[:window//2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.log10(t[1:]), np.log10(squared_fft[1:]))
    fig.savefig("foo.png")


if __name__ == "__main__":
    start = int(4e6) + 78000
    stop = start + 820000
    data = get_data(full=True, start=start, stop=stop)
    print(data.mean(1))
    data -= data.mean(1)[:, None]

    data = data[0, :]
    multiple_fft(data)
