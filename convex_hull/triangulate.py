import pymesh
import numpy as np

points = np.load("surface_points.npy")
simplices = np.load("surface_simplices.npy")

brain = pymesh.form_mesh(points, simplices)
pymesh.save_mesh("brain.stl", brain)
#hull = pymesh.load_mesh("test_remesh.stl")
