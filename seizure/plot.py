"""Plot state variables evaluaeted in a point."""

from pathlib import Path

import numpy as np

from postplot import plot_point_field

from postspec import PlotSpec

from postutils import (
    set_matplotlib_parameters,
    preprocess_wei,
)


set_matplotlib_parameters()

casedir = Path("test_pp_casedir")
times = np.load(casedir / "times.npy")

data_map = {
    name.split("-")[0]: np.load(casedir / name / f"probes_{name}.npy")
    for name in ("NKo-point", "NNao-point", "NClo-point", "Voli-point")
}

print(data_map)
processed = preprocess_wei(
    data_map,
    data_map.keys(),
    {"vol": 1.4368e-15, "beta0": 7},
    {"outdir": "."}
)

for data, spec in processed:
    plot_point_field(times, data, spec)
