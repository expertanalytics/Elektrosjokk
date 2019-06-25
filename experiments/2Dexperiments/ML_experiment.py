import resource
import warnings
import time
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

from postutils import (
    store_sourcefiles,
    simulation_directory,
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

from typing import Union


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


def get_post_processor(
        brain: xb.CardiacModel,
        outpath: Union[str, Path],
        dimension: int = 1
) -> Saver:
    """Create and return the post processor."""
    pp_spec = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(pp_spec)
    saver.store_mesh(brain.mesh, facet_domains=None)

    # field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=40)
    field_spec_checkpoint = FieldSpec(save_as=("xdmf"), stride_timestep=1)
    saver.add_field(Field("v", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("hdf5"), stride_timestep=40*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))

    points = get_points(dimension, num_points=10)
    point_field_spec = FieldSpec(stride_timestep=4)
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
    params = {
        "conductivity": conductivity,
        "Kinf_domain_size": Kinf_domain_size,
        "N": N,
        "dt": dt,
        "T": T,
        "dimension": dimension,
    }
    outdir = simulation_directory(parameters=params, directory_name=".simulations/cressman2D")
    print("----------------------")
    print(outdir)
    print("----------------------")
    store_sourcefiles(map(Path, ["ML_experiment.py", "experiment_utils.py"]), outdir)
    brain = get_brain(dimension, N, conductivity, Kinf_domain_size, K1, K2)
    solver = get_solver(brain)
    saver = get_post_processor(brain, outpath=outdir, dimension=dimension)

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
    return outdir

if __name__ == "__main__":
    # conductivity_list = (1/64, 1/8, 8)
    # KL_list = (1/2, 1/4,)
    # parameter_list = product(conductivity_list, KL_list)

    Mi = 1/64
    Cm = 1
    chi = 1.26e3
    lbda = Mi/2.76
    conductivity = Mi*lbda/(1 + lbda)*1/(Cm*chi)

    parameter_list = [
        (conductivity, 1/2, 100),
        (conductivity, 1/2, 200),
        (conductivity, 1/2, 300),
        (conductivity, 1/2, 400),
        (conductivity, 1/2, 500),
    ]

    def experiment(params, dt=0.05, T=1e0, dimension=2, K1=4, K2=8):
        """partial function wrapper."""
        conductivity, Kinf_domain_size, N = params
        kwargs = {
            "conductivity": conductivity,
            "Kinf_domain_size": Kinf_domain_size,
            "N": N,
            "dt": dt,
            "T": T,
            "K1": K1,
            "K2": K2,
            "dimension": dimension,
            "reload": False,
            "verbose": True
        }
        return run_ML_experiment(**kwargs)

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)

    tick = time.time()
    pool = Pool(processes=5)
    identifier_list = pool.map(experiment, parameter_list)
    tock = time.time()

    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    print("Execution time: {:.2f} s".format(tock - tick))

    # print("Making reports")
    # for casedir in identifier_list:
    #     make_report(casedir, dim=2)
    print("Sucess!")
