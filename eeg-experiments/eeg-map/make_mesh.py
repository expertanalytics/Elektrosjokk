import subprocess

import numpy as np

from pathlib import Path


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


if __name__ == "__main__":
    pos = get_positions()
    x, y, _ = pos.T

    # Define rectangle
    gmsh_text_file = "h = 1;\n"
    gmsh_text_file += "Point(1) = {-15, -10, 0, h};\n"
    gmsh_text_file += "Point(2) = {15, -10, 0, h};\n"
    gmsh_text_file += "Point(3) = {15, 10, 0, h};\n"
    gmsh_text_file += "Point(4) = {-15, 10, 0, h};\n"

    gmsh_text_file += "\n"
    gmsh_text_file += "Line(101) = {1, 2};\n"
    gmsh_text_file += "Line(102) = {2, 3};\n"
    gmsh_text_file += "Line(103) = {3, 4};\n"
    gmsh_text_file += "Line(104) = {4, 1};\n"
    gmsh_text_file += "Line Loop(201) = {101, 102, 103, 104};\n"
    gmsh_text_file += "Plane Surface(301) = {201};\n"

    # Add pritected points
    gmsh_text_file += "\n"
    point_counter = 10
    for xp, yp in zip(x, y):
        gmsh_text_file += "Point({point_id:d}) = {{{x:f}, {y:f}, 0, h}};\n".format(
            point_id=point_counter,
            x=xp,
            y=yp
        )
        point_counter += 1

    gmsh_text_file += "\n"
    for pid in range(10, point_counter):
        gmsh_text_file += "Point{{{:d}}} In Surface{{301}};\n".format(pid)

    with open("gmsh_meshing/protected_points.geo", "w") as of_handle:
        of_handle.write(gmsh_text_file)

    result = subprocess.check_output(["gmsh", "-2", "gmsh_meshing/protected_points.geo"])
    print(result.decode("utf-8"))

    result = subprocess.check_output([
        "meshio-convert",
        "gmsh_meshing/protected_points.msh",
        "gmsh_meshing/protected_points.xml"
    ])
    print(result.decode("utf-8"))


