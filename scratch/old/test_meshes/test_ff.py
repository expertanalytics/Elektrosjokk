from dolfin import *

mesh = UnitSquareMesh(10, 10)


class UpperCorner(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.5 and x[1] > 0.5 and on_boundary


class LowerCorner(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 and x[1] < 0.5 and on_boundary

ff = MeshFunction("size_t", mesh, 1)
ff.set_all(0)
LowerCorner().mark(ff, 1)
UpperCorner().mark(ff, 2)

File("ff.pvd") << ff
