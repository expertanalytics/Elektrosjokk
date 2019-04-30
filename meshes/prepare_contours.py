import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from svmtk import Surface

from scipy.spatial import cKDTree

import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def get_contour(surface, simplify_factor):
    slice = surface.slice(0, 0, 1, 0)
    slice.keep_component(0)
    slice.simplify(simplify_factor)

    contour = slice.get_constraints(0)
    return contour


def create_contour(contour):
    """Iterate through a contour by juming to the next unseen closes neighbour."""
    right_x = contour[:, 0] > -20       # TODO: parametrise
    lower_y = contour[:, 1] < 40        # TODO: parametrise
    contour = contour[~(right_x & lower_y)]

    # make unique x-values
    _, unique_x_idx = np.unique(contour[:, 0], return_index=True)
    contour = contour[unique_x_idx]

    # The point in the lower right of the polygon
    start_point = contour[:, 0].max(), contour[:, 1].min()
    contour = np.append(start_point, contour)
    contour.shape = (-1, 2)

    # the first poin in the polygon -> In the upper right
    start_index = np.argmax(contour[:, 1])

    # Indices of points not yet added to sorted contour
    remaining_indices = np.arange(contour.shape[0])

    # append starting point to sorted list of  indices
    sorted_point_indices = [start_index]
    remaining_indices[start_index] = -1     # set corresponding index to -1

    tree = cKDTree(contour)     # fast lookup of closes neighbours
    while len(sorted_point_indices) != contour.shape[0]:        # While there are points left
        distances, dist_indices = tree.query(
            contour[sorted_point_indices[-1]],
            k=len(sorted_point_indices) + 1
        )

        # close points not yet seen -- indices of dist_indices
        legal_indices = np.in1d(dist_indices, remaining_indices)
        min_legal_dist_idx = np.argmin(distances[legal_indices])

        # Next contour index
        next_index = dist_indices[legal_indices][min_legal_dist_idx]

        remaining_indices[next_index] = -1      # remove next_index from unseen pool
        sorted_point_indices.append(next_index)

    # print("start: ", contour[start_index])
    # print("stop: ", stop_point)
    sorted_slice = contour[sorted_point_indices]
    # sorted_slice = np.append(sorted_slice, stop_point)
    # sorted_slice.shape = (-1, 2)
    return sorted_slice


def plot_polygon(sorted_contour):
    """Plot a polygon from a contour generated by SVMTK slice.

    It is trial and error to get the slice looking nice.
    """

    fig, ax = plt.subplots(1)

    ax.set_xlim(-100, 0)
    ax.set_ylim(0, 100)
    for i in range(sorted_contour.shape[0] - 1):
        x0, y0 = sorted_contour[i, 0], sorted_contour[i, 1]
        x1, y1 = sorted_contour[i + 1, 0], sorted_contour[i + 1, 1]
        ax.plot((x0, x1), (y0, y1), color="r")

    x0, y0 = sorted_contour[-1, 0], sorted_contour[-1, 1]
    x1, y1 = sorted_contour[0, 0], sorted_contour[0, 1]
    ax.plot((x0, x1), (y0, y1), color="r")
    ax.grid(True)
    return fig


if __name__ == "__main__":
    surface_file_dir = Path("surface-files")

    surface_dict = {
        "skull": ("iss", 0.5),
        "pial": ("pial", 0.5),
        "white": ("white", 0.5)
    }

    # name = "skull"
    # name = "pial"
    name = "white"
    surf = Surface(str(surface_file_dir / f"{surface_dict[name][0]}_sclipped.off"))
    contour = get_contour(surf, surface_dict[name][1])
    print(contour.shape)

    sorted_slice = create_contour(contour)
    myfig = plot_polygon(sorted_slice)
    myfig.savefig("foo.png")

    np.save(f"{name}", sorted_slice)
