"""Plot a map of EEG activity."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from typing import List
from eegutils import read_zhe_eeg

from matplotlib.collections import PatchCollection



def get_positions(ect_dir_path: str="Documents/ECT-data") -> np.ndarray:
    """Read the electrode positions, and return a 2d array of positions.

    Arguments:
        ect_dir_path: Path to the ECT-data directory.
    """
    _ect_dir_path = Path(ect_dir_path)
    data = np.loadtxt(
        Path.home() / _ect_dir_path / "zhi/channel.pos",
        delimiter=",",
        usecols=(2, 3, 4)
    )
    return data


def get_eeg_value() -> List[float]:
    # Rather than use a class, I can use the send/recv feature of generators

    all_channels_seizure = read_zhe_eeg(
        start=4078000,
        stop=4902000,
        full=True
    )
    return all_channels_seizure


def save_frame(x, y, z, timestamp, vmin=-400, vmax=1200):
    fig, ax = plt.subplots(1)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])

    # norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    # color_mapper = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    color_mapper = plt.cm.ScalarMappable(norm=None, cmap="viridis")
    print(z.max(), z.min())
    z_colors = color_mapper.to_rgba(z)

    patches = []
    radius = 0.5
    for i in range(x.size):
        circle = plt.Circle((x[i], y[i]), radius, color=z_colors[i])
        patches.append(circle)
        ax.add_artist(circle)


    ###############################################################################################
    xy_points = np.loadtxt("head_outline.csv", delimiter=",", usecols=(0, 1), skiprows=1)
    xy_points /= 10     # Scale to cm
    ###############################################################################################

    ###############################################################################################
    centroid = np.sum(xy_points, axis=0)/xy_points.shape[0]
    sort_indices = np.argsort(np.arctan2(
        xy_points[:, 1] - centroid[1],
        xy_points[:, 0] - centroid[0]
    ))
    xy_points = xy_points[sort_indices]
    ###############################################################################################

    ###############################################################################################
    poly = plt.Polygon(xy_points, closed=True)

    # TODO: Make polygon a separate patch collection
    patch_collection_poly = PatchCollection([poly], alpha=0.3, color="tab:gray")
    ax.add_collection(patch_collection_poly)

    patch_collection_circles = PatchCollection(patches, alpha=1)
    # patch_collection_circles.set(array=np.arange(vmin, vmax, 10), cmap="viridis")
    # fig.colorbar(patch_collection_circles)
    # ax.add_collection(patch_collection_circles)
    ###############################################################################################
    fig.savefig("figures/eeg{:04d}.png".format(timestamp))


if __name__ == "__main__":
    positions = get_positions()

    x = positions[:, 0]
    y = positions[:, 1]

    all_eeg_channels = get_eeg_value()
    for i in range(1000):
        time = 5*i     # every 10th ms
        print(time)
        z = all_eeg_channels[:, time]
        save_frame(x, y, z, time)
