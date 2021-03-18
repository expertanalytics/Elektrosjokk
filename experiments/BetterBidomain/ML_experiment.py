import resource
import warnings
import time
import shutil

from make_report import make_report

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

from experiment_utils import (
    get_points,
    assign_initial_conditions,
    reload_initial_condition
)


warnings.simplefilter("ignore", UserWarning)


def get_post_processor(brain: xb.CardiacModel, outpath: Path) -> Saver:
    pp_spec = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(pp_spec)
    saver.store_mesh(brain.mesh)

    space_stride = 4
    field_spec = FieldSpec(
        save_as=("xdmf", "hdf5"),
        stride_timestep=space_stride
    )
    saver.add_field(Field("v", field_spec))
    saver.add_field(Field("vs", field_spec))

    points = get_points(brain.mesh.geometry().dim(), num_points=11)

    time_stride = 400
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
    """
    print("Conductivity: {:4.2f}, KL: {:4.2f}".format(conductivity, Kinf_domain_size))

    brain = get_brain(dimension, N, conductivity, Kinf_domain_size, K1, K2)
    solver = get_solver(brain)
    saver = get_post_processor(brain, outpath=Path(outpath))
    shutil.copyfile(__file__, Path(outpath) / Path(__file__).name)

    if reload:
        reload_initial_condition(solver, outpath)
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
    conductivity_list = (1.0,)
    KL_list = (1/4,)
    parameter_list = product(conductivity_list, KL_list)

    def experiment(params, N=500, dt=0.025, T=5e2, dimension=1, K1=4, K2=8):
        """partial function wrapper."""
        conductivity, Kinf_domain_size = params
        # identifier = "M{:3}-L{:3}".format(
        #     str(conductivity).replace(".", ""), str(Kinf_domain_size).replace(".", "")
        # )
        # outpath = Path("experimentI") / identifier
        outpath = "BetterBidomain"
        args = (conductivity, Kinf_domain_size, N, dt, T, K1, K2, dimension, outpath)
        kwargs = {"reload": False}
        return run_ML_experiment(*args, **kwargs)
    tick = time.time()
    experiment((1/8, 1/5))
    tock = time.time()
    print("Execution time: {:.2f} s".format(tock - tick))
    print("Success!")

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)
    """
    tick = time.time()
    pool = Pool(processes=4)
    identifier_list = pool.map(experiment, parameter_list)
    tock = time.time()
    """

    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    print("Execution time: {:.2f} s".format(tock - tick))

    print("Making reports")
    casedir = Path("BetterBidomain")
    make_report(casedir, dim=1)
    print("Sucess!")
