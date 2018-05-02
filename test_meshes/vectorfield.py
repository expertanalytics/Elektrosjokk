from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np


fig = plt.figure()
ax = fig.gca()

x, y = np.meshgrid(
    np.arange(0, 1, 0.1),
    np.arange(0, 1, 0.1),
)

u = y
v = 1 - x

ax.quiver(x, y, u, v)
plt.show()
input()
