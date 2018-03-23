"""Create plots from preprocessed ODE solutions."""

import logging
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

from typing import (
    Any,
    Tuple,
)

from ect.specs import Plot_spec


logger = logging.getLogger(name=__name__)


def plot_cell_model(
        plot_spec: Plot_spec,
        time: np.ndarray,
        outdir: str,
        save_format="png"
) -> None:
    """Iterate over a generator and create plots.

    Create a plot from each line in data_spec_tuple.
    """
    # Create figure and axis
    fig = plt.figure(figsize=(14, 14))
    ax1 = fig.add_subplot(111)
    ax1.grid(True)  # ???
    fig.suptitle(plot_spec.title, size=52)

    # if there are more legends
    for data, label in tuple(plot_spec.line):
        ax1.plot(time, data, label=label, linewidth=4)

    ax1.set_ylabel(plot_spec.ylabel, size=48)
    ax1.set_xlabel("Time (s)", size=48)

    ax1.legend(loc="best", fontsize=22)

    # Make the plot square
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect((x1 - x0)/(y1 - y0))

    # Update labelsize
    ax1.tick_params(axis="both", which="major", labelsize=28)
    ax1.tick_params(axis="both", which="minor", labelsize=28)

    # Set font size for the scientific axis scale
    tx = ax1.xaxis.get_offset_text()
    ty = ax1.yaxis.get_offset_text()
    tx.set_fontsize(28)
    ty.set_fontsize(28)
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{outdir}/{plot_spec.name}.{save_format}")
    plt.close(fig)
