import numpy as np

import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, 20)
x = np.sin(t)
y = np.cos(t)

plt.plot(x, y)
# plt.savefig("circle.png")

xy = np.dstack((x, y))[0]

EEG_points = [(0.0, 1.5), (1.5, 0.0), (-1.5, 0.0), (-1.5, 0.0)]

my_point = xy[3]
for p in EEG_points:
    r2 = np.linalg.norm(my_point - p)
    print(r2)


x = np.array([[0, 0],
              [1, 0],
              [1, 1],
              [0, 1]])


P = array([[ 0.25,  1.5 ],
           [ 1.5 ,  0.75],
           [ 0.75, -0.3 ]])

dist_matrix = x[:, None] - P[None, :]
distances = np.sqrt(np.power(dist_matrix, 2).sum(axis = -1))
inv_dist = 1/distances
inv_dist_norm = inv_dist/inv_dist.sum(-1)[:, None]

