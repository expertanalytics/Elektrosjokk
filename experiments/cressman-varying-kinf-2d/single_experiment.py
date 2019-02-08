import time
import shutil
import tqdm

import numpy as np
import dolfin as df

from pathlib import Path

from xalbrain import CardiacModel

from xalbrain.cellmodels import Cressman

from postfields import (
    Field,
    PointField,
)

from postspec import (
    FieldSpec,
    SaverSpec,
)

from post import Saver

from experiment_utils import (
    get_solver,
    get_outpath,
    assign_initial_conditions,
    reload_initial_condition,
    get_mesh,
    get_conductivities,
)

from typing import Union


def get_Kinf() -> df.Expression:
    """Return the varying Kinf."""

    kinf_string = """
        values[0] = 4;
        if (x[0] > 0.2 and x[0] < 0.4 and x[1] > 0.6 and x[1] < 0.8) {
            values[0] = 8;
        }

        if (x[0] > 0.6 and x[0] < 0.8 and x[1] > 0.6 and x[1] < 0.8) {
            values[0] = 9.5;
        }

        if (x[0] > 0.4 and x[0] < 0.6 and x[1] > 0.2 and x[1] < 0.4) {
            values[0] = 7;
        }
    """
    Kinf = df.Expression(kinf_string, degree=1)
    return Kinf


def get_points(dimension=1):
    if dimension == 1:
        _npj = 21*1j
        numbers = np.mgrid[0:1:_npj]
        return np.vstack(map(lambda x: x.ravel(), numbers)).reshape(-1, dimension)
    if dimension == 2:
        my_range = np.arange(10)/10

        foo = np.zeros(shape=(10, 2))
        foo[:, 0] = my_range

        bar = np.zeros(shape=(10, 2))
        bar[:, 0] = my_range
        bar[:, 1] = my_range
        return np.vstack((foo, bar))


def get_brain(
        dimension: int,
        N: int,
        conductivity: float
) -> CardiacModel:
    """
    Create container class for splitting solver parameters

    Arguments:
        dimension: The topological dimension of the mesh.
        N: Mesh resolution.
        conductivity: The conductivity, or rather, the conductivity times a factor.
        Kinf_domain_size: The side length of the domain where Kinf = 8. In 1D, this is
            simply the lengt of an interval.
    """
    mesh = get_mesh(dimension, N)
    Mi = get_conductivities(conductivity)
    time_const = df.Constant(0)
    Kinf = get_Kinf()

    model_parameters = Cressman.default_parameters()
    model_parameters["Koinf"] = Kinf
    model = Cressman(params=model_parameters)
    brain = CardiacModel(
        mesh,
        time_const,
        M_i=Mi,
        M_e=None,
        cell_models=model,
    )
    return brain


def get_post_processor(
        brain: CardiacModel,
        outpath: Union[str, Path],
        suffix: str = None,
        home: bool = False,
        dimension: int = 1
) -> Saver:
    """Create and return the post processor."""
    _outpath = Path(outpath)
    if suffix is None:
        suffix = time.strftime("%Y%m%d-%H%M%S")
    _outpath /= suffix

    if home:
        _outpath = Path.home() / _outpath

    pp_spec = SaverSpec(casedir=_outpath)
    saver = Saver(pp_spec)
    saver.store_mesh(brain.mesh, facet_domains=None)

    stride_timestep = 4
    field_spec = FieldSpec(save_as=("hdf5", "xdmf"), stride_timestep=stride_timestep)
    saver.add_field(Field("v", field_spec))

    points = get_points(dimension)

    point_field_spec = FieldSpec(stride_timestep=4)
    saver.add_field(PointField("point_v", point_field_spec, points))
    return saver


def run_ML_experiment(
        conductivity: float,
        Kinf_domain_size: float,
        N: int,
        dt: float,
        T: float,
        dimension: int,
        verbose=False,
        reload=False
) -> str:
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

    brain = get_brain(dimension, N, conductivity)
    solver = get_solver(brain, ode_dt=1)

    outpath = get_outpath(dimension)
    identifier = "foobar"

    if reload:
        reload_initial_condition(solver, outpath / identifier)
    else:
        assign_initial_conditions(solver)

    saver = get_post_processor(
        brain,
        outpath=outpath,
        suffix=identifier,
        dimension=dimension
    )
    # shutil.copyfile(__file__, outpath / identifier / Path(__file__).name)
    tick = time.clock()
    for i, ((t0, t1), (vs_, vs, vur)) in tqdm.tqdm(enumerate(solver.solve((0, T), dt)), total=T/dt - 1):
        if verbose:
            print("Timetep: {:d}".format(i))

        update_dict = {
            "v": vur,
            "point_v": vur,
        }

        saver.update(
            brain.time,
            i,
            update_dict
        )
        tock = time.clock()
        print("Solver time: ", tock - tick)
        tick = tock
    saver.close()       # TODO: Add context handler?
    return identifier


if __name__ == "__main__":
    tick = time.clock()
    run_ML_experiment(
        conductivity=0.5,
        Kinf_domain_size=0.3,
        N=500,
        dt=0.025,
        T=1e3,
        dimension=2,
        verbose=True
    )
    tock = time.clock()
    print("Success! duration: {}".format(tock - tick))
