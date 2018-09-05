from dolfin import *

from conductivites import get_conductivities
from shock import get_shock


mesh = Mesh("merge.xml.gz")


class BDSolver:
    def __init__(self, mesh, dt):
        self._mesh = mesh
        self._dt = Constant(dt)
        self.time = Constant(0)

    def solution_fields():
        return self.UV
        # return self.v_, self.vur

    def _create_variational_forms(self):
        F_ele = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        W = FunctionSpace(self._mesh, MixedElement((F_ele, F_ele)))
        u, v = TrialFunctions(W)
        w, q = TestFunctions(W)

        # Re = FiniteElement("R", self._mesh.ufl_cell(), 0)
        # W = FunctionSpace(mesh, MixedElement((F_ele, F_ele, Re)))
        # u, v, l = TrialFunctions(W)
        # w, q, lam = TestFunctions(W)


        self.UV = Function(W)
        self.UV_ = Function(W)

        # Me, Mi = get_conductivities(3)
        Me, Mi = Constant(100), Constant(100)

        stimulus = Expression(
            "1e-4*exp(-0.003*pow(x[0]-10, 2))*exp(-0.003*pow(x[1]-68, 2))*exp(-0.003*pow(x[2]-32.0, 2))*exp(-t)",
            t=self.time,
        degree=1)

        # U_, V_, _ = split(self.UV)
        U_, V_ = split(self.UV)
        dtc = self._dt

        # G = u*w*dx + dtc*inner(Me*grad(u), grad(w))*dx
        # G += dtc*inner(Me*grad(v), grad(w))*dx
        # G += dtc*inner(Me*grad(u), grad(q))*dx
        # G += v*q*dx + dtc*inner(Mi*grad(v), grad(q))*dx

        G = (u - V_)/dtc*w*dx + inner(Me*grad(u), grad(w))*dx
        G += inner(Me*grad(v), grad(w))*dx
        G += inner(Me*grad(u), grad(q))*dx
        G += v*q*dx + inner(Mi*grad(v), grad(q))*dx
        # G += (lam*u + l*q)*dx
        G += stimulus*w*ds

        a, L = system(G)
        return a, L

    def _create_solver(self):
        self._lhs, self._rhs = self._create_variational_forms()

        self._lhs_matrix = assemble(self._lhs)
        self._rhs_vector = Vector(self._mesh.mpi_comm(), self._lhs_matrix.size(0))

        # What is this?
        # self._lhs_matrix.init_vector(self._rhs_vector, 0)

        self.linear_solver = PETScKrylovSolver("gmres", "petsc_amg")
        self.linear_solver.set_operator(self._lhs_matrix)

    def _step(self):
        # Solve problem

        assemble(self._rhs, tensor=self._rhs_vector)
        self._rhs_vector -= self._rhs_vector.sum()/self._rhs_vector.size()

        self.linear_solver.solve(
            self.UV.vector(),
            self._rhs_vector,
        )
        print("norm: ", self._rhs_vector.norm("l2"))

    def solve(self, t0, T):
        self._create_solver()
        ufile = File("results/u.pvd")
        vfile = File("results/v.pvd")

        t = t0
        while t <= T:
            t += dt
            self.time.assign(t)

            self._step()
            self.UV_.assign(self.UV) # update solution on previous time step with current solution 

            UE, V = self.UV.split()

            ufile << UE
            vfile << V


if __name__ == "__main__":
    mesh = Mesh("merge.xml.gz")
    dt = 0.001
    bdsolver = BDSolver(mesh, dt)
    bdsolver.solve(0, 1)
