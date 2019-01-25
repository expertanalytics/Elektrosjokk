from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch
from eegutils import read_zhe_eeg

from scipy.fftpack import hilbert

from pyhht import EMD
from pyhht.visualization import plot_imfs


short_data = read_zhe_eeg(full=False)
my_data = short_data[0, :]

nt = 2
my_data_2s = my_data[:nt*5000]
t = np.linspace(0, nt, my_data_2s.size)

decomposer = EMD(my_data_2s)
imfs = decomposer.decompose()
foo = plot_imfs(my_data_2s, imfs, t)
plt.savefig("foo.png")
