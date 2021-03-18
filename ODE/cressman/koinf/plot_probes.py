import numpy as np
from postspec import PlotSpec
from postplot import plot_point_field

from post import (
    read_point_values,
    read_point_metadata
)


path = "out_cressman/20181011-132108"
dt = 1e-2

for name in ("v", "m", "h", "n", "Ca",  "K", "Na"):
    data = read_point_values(path, name)
    _data = data[:]
    plot_spec = PlotSpec("foo", name, "test\_{}".format(name), "mV")
    plot_point_field(
        np.linspace(0, _data.size*dt, _data.size),
        _data,
        plot_spec
    )
