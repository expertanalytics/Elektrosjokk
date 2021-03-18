import dolfin as df
import time

from coupled_utils import get_mesh


from post import Saver, Loader
from postspec import SaverSpec, FieldSpec, LoaderSpec
from postfields import Field


def create_anisotropy():
    saver_parameters = SaverSpec(casedir="anisotropy", overwrite_casedir=True)
    saver = Saver(saver_parameters)
    field_spec = FieldSpec(save_as=("xdmf", "hdf5"))
    saver.add_field(Field("poisson", field_spec))
    saver.add_field(Field("grad_poisson", field_spec))
    saver.add_field(Field("normalised_grad_poisson", field_spec))
    _time = df.Constant(0)

    mesh, cell_function, interface_function = get_mesh("squiggly_meshes", "squiggly")
    submesh = df.SubMesh(mesh, cell_function, 1)
    saver.store_mesh(submesh)

    V = df.FunctionSpace(submesh, "CG", 1)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    f = df.Expression(
        "A*exp(-a*(pow(x[0] - x0, 2) + pow(x[1] - y0, 2)))",
        A=1000,
        a=100,
        x0=0,
        y0=0,
        degree=1
    )

    form = df.inner(df.grad(u), df.grad(v))*df.dx + f*v*df.dx

    a = df.lhs(form)
    L = df.rhs(form)

    A = df.assemble(a, keep_diagonal=True)
    b = df.assemble(L)

    solution_vector = df.Function(V)
    bcs = df.DirichletBC(V, df.Constant(0), df.DomainBoundary())
    bcs.apply(A)

    solver = df.KrylovSolver("cg", "petsc_amg")
    solver.set_operator(A)
    solver.solve(solution_vector.vector(), b)
    saver.update(_time, 0, {"poisson": solution_vector})

    V_grad = df.VectorFunctionSpace(submesh, "CG", 1)
    my_grad = df.project(df.grad(solution_vector), V_grad)

    saver.update(_time, 0, {"grad_poisson": my_grad})

    normalised = df.project(my_grad / df.sqrt(df.dot(my_grad, my_grad)))
    saver.update(_time, 0, {"normalised_grad_poisson": normalised})


def get_anisotropy(normalised_gradient, sigma_l, sigma_n):
    A = df.as_matrix([
        [normalised_gradient[0], normalised_gradient[1]],
        [-normalised_gradient[1], normalised_gradient[0]]
    ])
    # A = df.as_matrix([
    #     [-normalised_gradient[0], -normalised_gradient[1]],
    #     [normalised_gradient[1], -normalised_gradient[0]]
    # ])
    sigma_matrix = df.as_matrix([[sigma_l, 0], [0, sigma_n]])
    return A*sigma_matrix*A.T


if __name__ == "__main__":
    create_anisotropy()

    loader = Loader(LoaderSpec(casedir="anisotropy"))
    _, ngp = next(loader.load_field("normalised_grad_poisson", vector=True))
    anisotropy = get_anisotropy(ngp, 12, 23)
    print(anisotropy)
