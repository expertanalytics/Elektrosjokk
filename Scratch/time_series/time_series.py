from dolfin import *
import numpy as np
from scipy.interpolate import interp1d


def compute_point_weight(x, points_array):
    """Compute the normalised square distance between `x` and each point in `point_list`."""
    square_dist_matrix = np.sqrt(np.power(x - points_array, 2).sum(axis=-1))
    scaled_point_weight =  square_dist_matrix/square_dist_matrix.sum()
    return scaled_point_weight


T = 10      # End time
sample_points = np.array([(-0.25, 1.5), (2.5, 1.75), (1.1, -0.1)])
time_array = np.linspace(0, T, 1000)
V1 = np.sin(time_array)
V2 = np.cos(time_array)
V3 = 10*np.exp(-time_array)

# Define mesh, function space and test/trial functions
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)


class MyExpression(Expression):
    """Set value to the sum of time series scaled by a normalised distance."""

    def __init__(
            self,
            time_axis,
            time_series_list,
            time,
            point_list,
            **kwargs
    ):
        """Linear interpolation of each time series."""
        self.time_func_list = [
            interp1d(time_axis, ts) for ts in time_series_list
        ]
        self.time = time                # The current time
        self.point_list = point_list    # The point corresponding to each time series

    def eval(self, value, x):
        """Scale the value of the time series with a normalised weight."""
        val = sum([factor*tf(self.time)for factor, tf in zip(
                compute_point_weight(x, sample_points),
                self.time_func_list,
        )])
        value[0] = val

    def value_type(self):
        return (1,)


time = 0
my_expr = MyExpression(time_array, [V1, V2, V3], time, sample_points, degree=1)

A = inner(grad(u), grad(v))*dx
L = Constant(1)*v*ds
# L = my_expr*v*ds

bc = DirichletBC(V, my_expr, CompiledSubDomain("near(x[0], 0) && on_boundary"))

U_  = Function(V)

ofile = File("results/u.pvd")

dt = 0.1
t = 0
while t < T:
    my_expr.time = t
    solve(A == L, U_, bcs=bc)
    print(U_.vector().norm("l2"))
    ofile << U_
    t += dt
