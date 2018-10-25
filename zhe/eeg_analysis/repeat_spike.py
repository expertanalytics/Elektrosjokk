import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

from square_sin import compare_loglogfft


if __name__ == "__main__":
    data = np.load("midsection.npy")

    signal1 = data[12000:12400]
    first_idx = np.where(signal1 > signal1[-1])[0][0]
    signal1 = signal1[first_idx:]

    signal2 = data[14000:14550]
    first_idx = np.where(signal2 > signal2[-1])[0][0]
    signal2 = signal2[first_idx:]

    signal3 = data[16000:16475]
    first_idx = np.where(signal3 > signal3[-1])[0][0]
    signal3 = signal3[first_idx:]

    signal4 = data[45250:46250]
    first_idx = np.where(signal4 > signal4[-1])[0][0]
    signal4 = signal4[first_idx:]

    signal5 = data[70000:81400]
    first_idx = np.where(signal5 > signal5[-1])[0][0]
    signal5 = signal5[first_idx:]

    signal_list = [signal1, signal2, signal3, signal4, signal5]
    for i, signal in enumerate(signal_list):
        signal = np.tile(signal, 10)
        compare_loglogfft(
            [(signal, "repeated signal{}".format(i + 1))],
            filename="repeated_plots/repeated_fft{}".format(i + 1)
        )
