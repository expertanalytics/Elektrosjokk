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
    grid_points,
    circle_points,
)

from multiprocessing import Pool
from itertools import product
from pathlib import Path

from experiment_utils import (
    get_solver,
    get_brain,
    assign_initial_conditions,
    reload_initial_condition
)

from typing import Union


warnings.simplefilter("ignore", UserWarning)


def get_post_processor(
        brain: xb.CardiacModel,
        outpath: Union[str, Path],
) -> Saver:
    """Create and return the post processor."""
    pp_spec = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(pp_spec)
    saver.store_mesh(brain.mesh, facet_domains=None)

    # field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=40)
    field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=40)
    saver.add_field(Field("v", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("checkpoint", "hdf5"), stride_timestep=40)
    saver.add_field(Field("vs", field_spec_checkpoint))

    points = circle_points(radii=[0, 0.1, 0.2, 0.3], num_points=[1, 6, 18, 24], r0=(0.8, 0.8))
    point_field_spec = FieldSpec(stride_timestep=1)
    saver.add_field(PointField("psd_v", point_field_spec, points))

    points = grid_points(dimension=2, num_points=10)
    point_field_spec = FieldSpec(stride_timestep=1)
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
        verbose: bool = False,
        overwrite_data: bool = False
) -> Path:
    """
    Run the simulation and store the results.

    Arguments:
        dt: Pde time step.
        T: The equaitons ar solved in [0, T].
        conductivity: The conductivity, or rather, the conductivity times a factor.
        N: Parameter for the number of mesh points.
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
    }
    outdir = simulation_directory(
        parameters=params,
        directory_name=".simulations/cressman2DCSF",
        overwrite_data=overwrite_data
    )
    store_sourcefiles(map(Path, ["ML_experiment.py", "experiment_utils.py"]), outdir)

    brain = get_brain(
        mesh_resolution=N,
        conductivity=conductivity,
        kinf_domain_size=Kinf_domain_size,
        csf_start=0.75,
        K1=K1,
        K2=K2
    )
    solver = get_solver(brain=brain)
    saver = get_post_processor(brain, outpath=outdir)

    assign_initial_conditions(solver=solver)

    tick = time.time()
    for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
        tock = time.clock()
        if verbose:
            print("Timetep: {:d} --- {}".format(i, tock - tick))

        update_dict = {
            "v": vur,
            "vs": vs,
            "point_v": vur,
            "psd_v": vur
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
        # (conductivity, 1/2, 200),
        # (conductivity, 1/2, 300),
        # (conductivity, 1/2, 400),
        # (conductivity, 1/2, 500),
    ]

    def experiment(params, dt=0.05, T=5e2, K1=4, K2=8):
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
            "verbose": True,
            "overwrite_data": True
        }
        return run_ML_experiment(**kwargs)

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)

    tick = time.time()
    pool = Pool(processes=1)
    identifier_list = pool.map(experiment, parameter_list)
    tock = time.time()

    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    print("Execution time: {:.2f} s".format(tock - tick))
    print("Sucess!")
