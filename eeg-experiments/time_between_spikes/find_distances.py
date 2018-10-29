"""Compute the distance between the eeg electrodes."""

import numpy as np

from pathlib import Path

from typing import(
    Iterator,
    List
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
        dists = -2*np.dot(X, X.T).astype("f8")      # Need floats for sqrt later
        dists += np.sum(np.power(X, 2), axis=1)
        dists += np.sum(np.power(X, 2), axis=1)[:, None]
        np.sqrt(dists, out=dists)
        return dists


if __name__ == "__main__":
    data = get_positions()
    print(compute_distances(data))
