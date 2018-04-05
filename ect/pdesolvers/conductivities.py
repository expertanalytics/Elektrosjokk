import xalbrain as xb


def setup_conductivities(mesh):
    Q = xb.FunctionSpace(mesh, "DG", 0)
    Mi = xb.Function(Q)
    Me = xb.Function(Q)

    Mi.vector()[:] = 10
    Me.vector()[:] = 10
    return Mi, Me
