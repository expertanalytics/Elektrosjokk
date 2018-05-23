from dolfin import *


mesh = UnitSquareMesh(10, 10)


class Xdir(Expression):
    def eval(self, value, x):
        if x[1] >= 0.5:
            value[0] = x[1]*x[0]
            value[1] = 1.0 - x[0]
        else:
            value[0] = -x[1]*x[0]
            value[1] = x[0] - 1

    def value_shape(self):
        return (2,)

class Ydir(Expression):
    def eval(self, value, x):
        if x[1] >= 0.5:
            value[0] = 1.0 - x[0]
            value[1] = -x[1]*x[0]
        else:
            value[0] = -(x[0] - 1)
            value[1] = x[1]*x[0]

    def value_shape(self):
        return (2,)


Vv = VectorFunctionSpace(mesh, "CG", 1)
xfoo = project(Xdir(degree=1), Vv)
yfoo = project(Ydir(degree=1), Vv)

A = as_matrix([
    [xfoo[0], yfoo[0]],
    [xfoo[1], yfoo[1]]
])

g1 = Constant(1.0)
g2 = Constant(2.0)

M = diag(as_vector([g1, g2]))

foobar = A*M*A.T

xvec = as_vector([1, 0])
test = project(dot(xvec, foobar), Vv)
plot(test)
input()

yvec = as_vector([0, 1])
test = project(dot(yvec, foobar), Vv)
plot(test)
input()
