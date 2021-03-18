import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from analyse_eeg import get_data

from eegutils import get_default_filter

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def compute_psd(data, channels, fs=5000):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in channels:
        f, Pxx_den = signal.welch(data[i, :], fs, nperseg=256)
        # idx = f <= 1200
        idx = np.ones_like(f, dtype=bool)
        # plt.plot(f, Pxx_den)
        ax.plot(np.log10(f[1:]), 10*np.log10(Pxx_den[1:]), linewidth=1)     # What is this?

    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PDS [Bd/Hz]")
    fig.savefig("psd.png")

    Pxx, freqs = plt.psd(data[i, :], Fs=fs, NFFT=512)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.log10(freqs), np.log10(Pxx))
    fig.savefig("foo.png")


def loglogfft(data, channels, fs=5000):
    my_filter = get_default_filter()
    low_pass_data = my_filter(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for _data, label in zip((data, low_pass_data), ("raw", "smooth")):
        n = _data[0, :].size
        sp = np.fft.rfft(_data[0, :]).real[:n//2]
        t = np.fft.rfftfreq(n, d=1/fs)[:n//2]
        ax.plot(np.log10(t[1:]), np.log10((sp*sp)[1:]), label=label)

    # xtick_names = plt.setp(ax, xticklabels=10**ax.get_xticks())
    # plt.setp(xtick_names, rotation=0, fontsize=8)      # set property on firt argument

    # ytick_names = plt.setp(ax, yticklabels=10**ax.get_yticks())
    # plt.setp(ytick_names, rotation=0, fontsize=8)      # set property on firt argument

    # from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    # formatter = ScalarFormatter()

    # formatter = FormatStrFormatter("%1.1e")
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)
    # ax.ticklabel_format(axis="both", style="sci", useMathText=True)

    ax.legend()
    ax.set_ylabel(r"log($fft^2$)")
    ax.set_xlabel("log(frequencies) Hz")
    fig.savefig("spectrum_spikes.png")


def compare_loglogfft(data, ranges, fs=5000, filename=None):
    """No filter. ranges is in per second"""
    fig = plt.figure(figsize=(10, 10))
    subplot_idx = 1
    for start, stop in ranges:
        _data = data[start*fs:stop*fs]
        n = _data.size
        sp = np.fft.rfft(_data).real[:n//2]     # Try w/wo this
        sp_power = np.power(sp, 2)
        t = np.fft.rfftfreq(n, d=1/fs)[:n//2]

        ax = fig.add_subplot(2, 2, subplot_idx)
        subplot_idx += 1
        # ax.plot(np.log10(t[1:]), np.log10(sp_power[1:]))
        ax.plot(t, sp_power)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(r"{}s -- {}s".format(start, stop), fontsize=16)
        ax.set_xlabel(r"$Hz$", fontsize=12)
        ax.set_ylabel(r"$fft(eeg).real^2$", fontsize=12)
        ax.grid(True)

    if filename is None:
        filename = "fft_windows"
    fig.savefig("{}.png".format(filename))


if __name__ == "__main__":
    start = int(4e6) + 78000
    stop = start + 820000
    data = get_data(full=True, start=start, stop=stop)
    # data -= data.mean(1)[:, None]
    # my_filter = get_default_filter()
    # low_pass_data = my_filter(data)

    # compute_psd(low_pass_data, channels=[0,])
    # loglogfft(data, channels=[0,])
    compare_loglogfft(data[0, :], [(0, 1), (0, 2), (0, 4), (0, 8)])
