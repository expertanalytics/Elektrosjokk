import numpy as np
from prepare_contours import plot_polygon
from pathlib import Path
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev


if __name__ == "__main__":
    contour = np.load("skull.npy")
    # contour = np.load("pial.npy")
    # contour = np.load("white.npy")

    print(contour.shape)

    ymax_xval = contour[np.argmax(contour[:, 1])][0]
    contour = contour[contour[:, 0] < ymax_xval]

    fig, ax = plt.subplots(1)

    x = contour[:, 0]
    y = contour[:, 1]

    tck, u = interp.splprep([x, y], s=2.0)
    new_x, new_y = interp.splev(u, tck)

    ax.plot(x, y, 'ro')
    ax.plot(new_x, new_y, 'r-')
    plt.show()
