from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np

x, y = np.meshgrid(
    np.arange(0, 1.1, 0.1),
    np.arange(0, 1.1, 0.1),
)
# u = (y - 0.5)*(1*(x > 0.5) - 1*(x < 0.5))
# v = (1 - 4*(x - 0.5)**2)*(y > 0.5)

# u = (0.5 - y)*(1*(x > 0.5) - 1*(x < 0.5))
# v = -(1 - 4*(x - 0.5)**2)*(y < 0.5)

u = (x - 0.5)*y
v = 1 - y

fig = plt.figure()
ax = fig.gca()

ax.quiver(x, y, u, v)
plt.show()
input()
