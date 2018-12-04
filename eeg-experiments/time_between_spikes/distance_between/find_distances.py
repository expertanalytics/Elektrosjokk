"""Compute the distance between the eeg electrodes."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.spatial.distance import (
    pdist,
    squareform,
)

from scipy.spatial import cKDTree

from typing import (
    Iterator,
    List,
)


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


def compute_distances(X: np.ndarray) -> np.ndarray:
        """Compute the distance between the rows in X. This function computes (a^2 - 2ab + b^2)."""
        return squareform(pdist(X, metric="euclidean"))


def plot_hist_dist():
    data = get_positions()
    dists = compute_distances(data)
    assert np.allclose(dists, dists.T, atol=1e-8)

    # Get all unique distances
    unique_dists = np.unique(dists[dists > 0].ravel())
    print("min: ", unique_dists.min())
    print("max: ", unique_dists.max())
    print("mean: ", unique_dists.mean())

    ax = sns.distplot(
        unique_dists,
        hist=True,
        kde=True,
        color='darkblue',
        hist_kws={'edgecolor': 'black'},
        kde_kws={'linewidth': 4}
    )
    fig = ax.get_figure()
    fig.savefig("dist_hist.png")
    plt.close(fig)


def get_pairs(data, r=9):
    tree = cKDTree(data)        # Use r = 9cm
    point_pairs = tree.query_pairs(r=9)
    return point_pairs


def get_knearest(data, points, k=1):
    tree = cKDTree(data)
    point_pairs = tree.query(points, k=k)
    return point_pairs


def compute_distances_loop(data, points):
    distance_list = []
    for i, j in points:
        distance = np.linalg.norm(data[i] - data[j])
        assert distance < 9
        distance_list.append(distance)
    return np.asarray(distance_list)


if __name__ == "__main__":
    data = get_positions()
    points = get_pairs(data)
    distances = compute_distances_loop(data, points)
    print(points)
    print(distances)
