import pandas as pd
import xalbrain as xb
import numpy as np

from ect.specs import (
    PlotSpec,
)

from ect.plotting import (
    plot_line,
    plot_multiple_lines,
)

test_plot_spec = PlotSpec(outdir="figures", name="test", title="V", ylabel=r"mV")
plot_spec = PlotSpec(outdir="figures", name="comparison", title="V", ylabel=r"mV")

params = xb.cellmodels.Wei.default_parameters()
df0 = pd.read_pickle("theta05.pkl")
time = df0["time"].values
data0 = df0["V"].values

# df1 = pd.read_pickle("theta05_coarse.pkl")
# data1 = df1["V"].values

df1 = pd.read_pickle("theta0.pkl")
# time = df1["time"].values
data1 = df1["V"].values

df2 = pd.read_pickle("theta1.pkl")
data2 = df2["V"].values

df3 = pd.read_pickle("ode.pkl")
data3 = df3["V"].values

df4 = pd.read_pickle("numba_ode.pkl")
time4 = df4["time"].values
data4 = df4["V"].values

df5 = pd.read_pickle("numba_ode05.pkl")
data5 = df5["V"].values

df6 = pd.read_pickle("ode05.pkl")
data6 = df6["V"].values

plot_line(time4, data4, test_plot_spec)
# assert False, (_data0.shape, _data1.shape)

import math
l1 = 0.5*(3 + math.sqrt(5))
l2 = 0.5*(3 - math.sqrt(5))
c1 = 0.5*(1 + math.sqrt(5))
c2 = 0.5*(1 - math.sqrt(5))
x = c1*np.exp(l1*(time + 1e-2)) + c2*np.exp(l2*(time + 1e-2))

# assert False, (data0[::100].shape, data1[::1].shape, data2[::1].shape, data3[::1].shape, data4[::1000].shape)
# assert False, (time.shape, data0.shape, data2.shape, data3[::1000].shape)
plot_multiple_lines(
    time[::1],
    # {"fine": _data0, "coarse": data1},
    {"theta05": data0[::1], "theta1": data2[::1], "ode": data3[::1000]},#, "theta0": data1[::1]},# },# "numba": data4[::1000], "numba05": data5[::1000], "ode05": data6[::1]},
    plot_spec
)




# time: np.ndarray,
# data: np.ndarray,
# plot_spec: PlotSpec,
