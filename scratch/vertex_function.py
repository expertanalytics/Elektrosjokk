from dolfin import *

mesh = UnitSquareMesh(10, 10)

vertex_function = MeshFunction("size_t", mesh, 0)
vertex_function.set_all(0)

CompiledSubDomain("x[0] >= 0.5").mark(vertex_function, 1)


import numpy as np
