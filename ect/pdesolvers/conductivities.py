import xalbrain as xb


def setup_conductivities(mesh):
    Q = xb.FunctionSpace(mesh, "CG", 1)
    Mi = xb.Function(Q)
    Me = xb.Function(Q)

    Mi.vector()[:] = 0.1
    Me.vector()[:] = 0.3
    return Mi, Me
