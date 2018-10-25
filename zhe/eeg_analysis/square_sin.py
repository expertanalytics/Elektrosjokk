import numpy as np
import matplotlib.pyplot as plt

from postutils import square_pulse


def square_and_sin(frequency, T=1):
    time = np.linspace(0, T, 2000)              # seconds
    squareval = np.vectorize(square_pulse)(time, 4e-2, 10, 1)

    sinval = np.zeros_like(time)
    for frequency in range(1, 21, 1):
        sinval += np.sin(2*np.pi*frequency*time)
    sinval /= sinval.max()
    return time, sinval, squareval


def compare_loglogfft(data_list, fs=5000, filename=None):
    """No filter. ranges is in per second"""
    fig = plt.figure(figsize=(10, 10))
    subplot_idx = 1
    for _data, data_title in data_list:
        n = _data.size
        sp = np.fft.rfft(_data).real[:n//2]     # Try w/wo this
        sp_power = np.power(sp, 2)
        t = np.fft.rfftfreq(n, d=1/fs)[:n//2]

        ax = fig.add_subplot(len(data_list), 1, subplot_idx)
        subplot_idx += 1
        # ax.plot(np.log10(t[1:]), np.log10(sp_power[1:]))
        ax.plot(t, sp_power)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(data_title, fontsize=16)
        ax.set_xlabel(r"$Hz$", fontsize=12)
        ax.set_ylabel(r"$fft(eeg).real^2$", fontsize=12)
        ax.grid(True)

    if filename is None:
        filename = "fft_windows"
    fig.savefig("{}.png".format(filename))

if __name__ == "__main__":
    t, y1, y2 = square_and_sin(10)

    plt.plot(t, y1, t, y2)
    plt.savefig("square_sin.png")

    compare_loglogfft([(y1, "sin, $f=1 \dots 20$"), (y2, "square, $f = 10$")], filename="square_sin")
