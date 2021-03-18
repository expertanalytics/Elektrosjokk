import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from boxplot_eeg import get_data

import scipy.signal as sig


def butter_lowpass_filter(data, cutoff, fs=5000, order=8):

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.filtfilt(b, a, data)
    return y

def find_peaks_v2(data, name=None):
    time = np.linspace(0, data.size/5000, data.size)        # in ms
    low_pass_data = butter_lowpass_filter(data, 20, order=5)

    peaks, properties = sig.find_peaks(
        low_pass_data,
        prominence=10,
        width=25        # TODO: Is this right?
    )


    if name is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, data, label="raw eeg")
        ax.plot(time, low_pass_data, alpha=0.5, linestyle="-.", label="smooth (low pass 20 Hz)")

        # ax.plot(time, savgol_data)
        print("plotting lines")
        # for p in peaks:
        ax.plot(time[peaks], data[peaks], "o", color="r", alpha=0.75)
        """
        # ax.vlines(
        #     x=time[peaks],
        #     ymin=data[peaks] - properties["prominences"],
        #     ymax=data[peaks],
        #     color="C1"
        # )
        ax.hlines(
            y=properties["width_heights"],
            xmin=properties["left_ips"]/5000,
            xmax=properties["right_ips"]/5000,
            color="C1"
        )
        """

        ax.set_title("Peaks in the EEG signal", fontsize=20)
        ax.set_ylabel(r"$\mu V$")
        ax.set_xlabel("Time [s]")
        ax.yaxis.grid(True)
        ax.legend(fontsize=12)

        # fig.savefig("check_eeg_peakfinding/{}.png".format(name))
        fig.savefig("{}.png".format(name))
    return peaks, properties


if __name__ == "__main__":
    start = int(4e6)
    data = get_data(full=True)[:, start + 78000:start + 820000]
    print(data.shape)
    # print(data.shape)
    # plot_data("channel1_stimulus_seizure_supression", data, start=int(4e6), N=1, lines=[75000, 820000])
    # plot_data("channel1_stimulus", data, start=start, stop=start + 7500)
    # plot_data("channel1_seizure", data, start=start + 7500, stop=start + 820000)
    # plot_data("channel1_supression", data, start=start + 820000)

    # find_maxima(data[start + 75000: start + 820000])
    # find_maxima(data[:start + 75000])
    # plot_data("bar", data[start:start + 50000])

    # midsection = np.save("midsectioon.npy", data[start + 78000: start + 822000])
    # midsection = np.load("midsectioon.npy")


    for i in (0, 1, 14, 24, 32, 56):
        midsection = data[i]
        peaks, properties = find_peaks_v2(midsection, "eeg_spikes_{}".format(i))

        dp = np.diff(peaks)/5
        print(i, dp.mean(), dp.std())
        time = np.linspace(0, midsection.size/5000, dp.size)

        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(111)
        ax.plot(time, dp)
        ax.set_title("Time between spikes for the duration of the seizure", fontsize=32)
        ax.set_xlabel("Time [s]", fontsize=20)
        ax.set_ylabel(r"$\Delta t$ [ms]", fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=12)

        fig.savefig("time_between_spikes.png")
        del fig
        break

    # batch = 5*5000
    # i = 0
    # while (i + 1)*batch < midsection.size:
    #     print(midsection.size, i*batch)
    #     find_peaks_v2(midsection[i*batch: (i + 1)*batch], "peaks{}".format(i))
    #     i += 1
