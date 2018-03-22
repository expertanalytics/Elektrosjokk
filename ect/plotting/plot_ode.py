"""Create plots from preprocessed ODE solutions."""

import logging
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

from typing import (
    Any,
    Tuple,
)


logger = logging.getLogger(name=__name__)


def plot_cell_model(
        time: np.ndarray,
        data_spec_tuple: Tuple[Any],
        outdir: str,
        save_format="png"
) -> None:
    """Iterate over a generator and create plots.

    Create a plot from each line in data_spec_tuple.
    """
    _data_spec_tuple = tuple(data_spec_tuple)
    for data_spec in data_spec_tuple:
        # Create figure and axis
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(111)
        fig.suptitle(data_spec.title, size=52)
    
        # Plot the data
        ax1.plot(time, data_spec.data, label=data_spec.label, linewidth=4)
        ax1.set_ylabel(data_spec.ylabel, size=48)
        ax1.set_xlabel("Time (s)", size=48)
        ax1.grid(True)  # ???
        ax1.legend(loc="best")

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

        os.makedirs(f"{outdir}")
        fig.savefig(f"{outdir}/{name}.{save_format}")
        plt.close(fig)
