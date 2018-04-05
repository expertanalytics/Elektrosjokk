"""Create plots from preprocessed ODE solutions."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

from typing import (
    Any,
    Tuple,
    Dict,
)

from ect.specs import PlotSpec


logger = logging.getLogger(name=__name__)


def plot_multiple_lines(
        time: np.ndarray,
        data_dict: Dict[str, np.ndarray],
        plot_spec: PlotSpec
) -> None:
    _plot_multiple_lines(time, data_dict, plot_spec)


def plot_line(
        time: np.ndarray,
        data: np.ndarray,
        plot_spec: PlotSpec,
) -> None:
    _plot_multiple_lines(time, {plot_spec.name: data}, plot_spec)


def _plot_multiple_lines(
        time: np.ndarray, 
        data_dict: Dict[str, np.ndarray],
        plot_spec: PlotSpec,
        check_time: bool = False
) -> None:
    fig = plt.figure(figsize=plot_spec.figsize)
    ax1 = fig.add_subplot(111)
    ax1.grid(plot_spec.grid)  # TODO: ???
    fig.suptitle(plot_spec.title, size=52)

    # if there are more legends
    for label, data in data_dict.items():
        ax1.plot(time, data, label=label, linewidth=plot_spec.line_width)

    ax1.set_ylabel(plot_spec.ylabel, size=48)
    ax1.set_xlabel("Time (s)", size=48)

    ax1.legend(loc=plot_spec.label_loc, fontsize=22)

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
    Path(f"{plot_spec.outdir}").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{plot_spec.outdir}/{plot_spec.name}.{plot_spec.save_format}")
    plt.close(fig)


def plot_cell_model(
        plot_spec: PlotSpec,
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
