import resource
import warnings
import time
import shutil 
import dolfin as df
import numpy as np
import xalbrain as xb

from postfields import (
    Field,
    PointField,
)

from postspec import (
    FieldSpec,
    SaverSpec,
    LoaderSpec,
)

from post import (
    Saver,
    Loader,
)

from multiprocessing import Pool
from itertools import product
from pathlib import Path

from experiment_utils import (
    get_solver,
    get_brain,
)

from make_report import make_report

from experiment_utils import (
    get_points,
    assign_initial_conditions,
    assign_custom_initial_conditions,
    reload_initial_condition
)


warnings.simplefilter("ignore", UserWarning)


class CustomInitialCondition:

    def __init__(self, L, ic_k4, ic_k8, **kwargs):
        self.d1 = 0.5 - L/2
        self.d2 = 0.5 + L/2
        self.ic_k4 = ic_k4
        self.ic_k8 = ic_k8
        super().__init__(**kwargs)

    def eval(self, values, x):
        d1 = self.d1
        d2 = self.d2
        ic_k4 = self.ic_k4
        ic_k8 = self.ic_k8

        if x[0] > d1 and x[0] < d2 and x[1] > d1 and x[1] < d2:
            values[:] = ic_k8     # type tuple = think
        else:
            values[:] = ic_k4

    def value_shape(self):
        return (9,)


def get_post_processor(brain: xb.CardiacModel, outpath: Path) -> Saver:
    pp_spec = SaverSpec(casedir=outpath)
    saver = Saver(pp_spec)
    saver.store_mesh(brain.mesh)

    space_stride = 4*100
    space_stride = 4
    field_spec = FieldSpec(
        save_as=("xdmf", "hdf5"),
        stride_timestep=space_stride
    )
    saver.add_field(Field("v", field_spec))
    saver.add_field(Field("vs", field_spec))

    points = get_points(brain.mesh.geometry().dim(), num_points=11)

    time_stride = 4
    point_field_spec = FieldSpec(stride_timestep=time_stride)
    saver.add_field(PointField("point_v", point_field_spec, points))
    return saver


def run_ML_experiment(
        conductivity: float,
        Kinf_domain_size: float,
        N: int,
        dt: float,
        T: float,
        K1: float,
        K2: float,
        dimension: int,
        outpath: Path,
        ode_step_fraction: int = 1,
        verbose: bool = False,
        reload: bool = False
) -> Path:
    """
    Run the simulation and store the results.

    Arguments:
        dt: Pde time step.
        T: The equaitons ar solved in [0, T].
        conductivity: The conductivity, or rather, the conductivity times a factor.
        N: Parameter for the number of mesh points.
        dimension: The topological dimension of the mesh.
        Kinf_domain_size: The side length of the domain where Kinf = 8. In 1D, this is
            simply the lengt of an interval.
        ode_step_raction: The number of ode time steps per PDE time step.
    """
    print("Conductivity: {:4.2f}, KL: {:4.2f}".format(conductivity, Kinf_domain_size))

    brain = get_brain(dimension, N, conductivity, Kinf_domain_size, K1, K2)
    solver = get_solver(brain, ode_dt=dt/ode_step_fraction)
    saver = get_post_processor(brain, outpath=Path(outpath))
    shutil.copyfile(__file__, Path(outpath) / Path(__file__).name)

    if reload:
        ic_k4 = [               # t = 25460.837909645645
            -6.66931519e+01,
            1.23780459e-02,
            7.22412631e-02,
            9.77018443e-01,
            1.70927726e-07,
            3.99649516e+00,
            1.70943579e+01
        ]
        ic_k8 = [               # t = 32164.75743957227
            -5.82101937e+01,
            3.60140064e-02,
            1.36508955e-01,
            9.23030799e-01,
            4.83351211e-06,
            7.03563274e+00,
            1.63702641e+01
        ]
        custom_ic = CustomInitialCondition(Kinf_domain_size, ic_k4, ic_k8)
        assign_custom_initial_conditions(solver, custom_ic)
    else:
        assign_initial_conditions(solver)

    tick = time.time()
    for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
        tock = time.clock()
        if verbose:
            print("Timetep: {:d} --- {}".format(i, tock - tick))

        update_dict = {
            "v": vur,
            "vs": vs,
            "point_v": vur,
        }
        saver.update(brain.time, i, update_dict)
        tick = tock
    saver.close()
    return outpath


if __name__ == "__main__":
    conductivity_list = (1/64, 1/8, 8)
    KL_list = (1/2, 1/4,)
    parameter_list = product(conductivity_list, KL_list)

    def experiment(params, N=500, dt=0.025, T=1e1, dimension=2, K1=4, K2=8):
        """partial function wrapper."""
        conductivity, Kinf_domain_size = params
        identifier = "M{}-L{}".format(
            str(conductivity).replace(".", "")[:3], str(Kinf_domain_size).replace(".", "")[:3]
        )
        outpath = Path("experimentI") / identifier
        args = (conductivity, Kinf_domain_size, N, dt, T, K1, K2, dimension, outpath)
        kwargs = {"reload": False}
        return run_ML_experiment(*args, **kwargs)

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)

    tick = time.time()
    pool = Pool(processes=6)
    identifier_list = pool.map(experiment, parameter_list)
    tock = time.time()

    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    print("Execution time: {:.2f} s".format(tock - tick))

    # print("Making reports")
    # for casedir in identifier_list:
    #     make_report(casedir, dim=2)
    print("Sucess!")
