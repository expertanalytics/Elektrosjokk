import pandas as pd
import xalbrain as xb
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

df0 = pd.read_pickle("theta05.pkl")
time05 = df0["time"].values
data0 = df0["V"].values

df1 = pd.read_pickle("theta0.pkl")
time = df1["time"].values
data1 = df1["V"].values

df2 = pd.read_pickle("theta1.pkl")
data2 = df2["V"].values

import math
l1 = 0.5*(3 + math.sqrt(5))
l2 = 0.5*(3 - math.sqrt(5))
c1 = 0.5*(1 + math.sqrt(5))
c2 = 0.5*(1 - math.sqrt(5))
x0 = c1*np.exp(l1*(time05 + 5e-3)) + c2*np.exp(l2*(time05 + 5e-3))
x1 = c1*np.exp(l1*(time)) + c2*np.exp(l2*(time))
# x1 = c1*np.exp(l1*(time + 1e-4)) + c2*np.exp(l2*(time + 1e-4))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(time05, data0, label="theta 0.5")
ax1.plot(time05, x0, label="exact 0.5")
ax1.legend()
ax1.grid()

ax2.plot(time, data1, label="theta 0")
ax2.plot(time, data2, label="theta 1")
ax2.plot(time, x1, label="exact")
ax2.grid()
ax2.legend()



fig.savefig("figures/new.png")
