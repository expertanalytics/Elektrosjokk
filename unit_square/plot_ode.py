import numpy as np

data = np.load("wei_solution.npy")

import matplotlib.pyplot as plt

t = np.linspace(0, 1, data.shape[0])

name_list = ["V", "m", "n", "h", "NKo", "NKi", "NNao", "NNai", "NClo", "NCli", "vol", "O"]

for i in range(data.shape[1]):
    fig, ax1 = plt.subplots(1)
    ax1.plot(t, data[:, i])
    fig.legend(name_list[i])
    fig.savefig(f"variable_{i}.png", label=name_list[i])
    plt.close()
input()
