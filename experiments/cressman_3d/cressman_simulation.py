import dolfin as df
import numpy as np
import pandas as pd
import xalbrain as xb

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

from xalbrain.cellmodels import Cressman, Wei

from postfields import (
    Field,
    PointField,
)

from postspec import (
    FieldSpec,
    PostProcessorSpec,
)

from stimulus import ECT_current

from post import Saver

from pathlib import Path
import time

from typing import (
    Tuple,
    Any,
    Dict,
)


def get_mesh() -> df.Mesh:
    """Create the mesh."""
    mesh = df.UnitCubeMesh(20, 20, 20)       # 1cm time 1cm
    mesh.coordinates()[:] /= 10
    return mesh


def get_conductivities() -> Tuple[Any, Any]:
    """Create the conductivities."""
    Mi = df.Constant(2)     # mS/cm
    Me = df.Constant(1)
    return Mi, Me


def get_facet_functions(mesh: df.Mesh) -> df.MeshFunction:
    ff = df.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    ff.set_all(0)
    df.CompiledSubDomain("near(x[0], 0) && on_boundary").mark(ff, 11)
    df.CompiledSubDomain("near(x[0], 0.1) && on_boundary").mark(ff, 21)
    return ff


def get_ect_current(
        time: df.Constant,
        interval: float,
        ff: df.MeshFunction,
        keys: Tuple[int, int]
) -> Dict[int, df.Expression]:
    """This turned out to be too fancy."""
    area_list = [
        df.assemble(         # Compute area
            df.Constant(1)*df.ds(
                domain=ff.mesh(),
                subdomain_data=ff,
                subdomain_id=i
            )
        ) for i in keys
    ]

    amplitude = 3/2   # mA/cm^2         # /2 because we have a positive and negative lead
    period = 1000   # ms
    duration = 600  # ms

    ect_current_dict = {
        0: df.Constant(0),
        keys[0]: ECT_current(time, interval, amplitude, period, duration, area_list[0], degree=1),
        keys[1]: -ECT_current(time, interval, amplitude, period, duration, area_list[1], degree=1)
    }
    return ect_current_dict


def get_brain(interval: float) -> CardiacModel:
    mesh = get_mesh()
    Mi, Me =  get_conductivities()
    time_const = df.Constant(0)
    ff = get_facet_functions(mesh)
    ect_current_dict = get_ect_current(time_const, interval, ff, (11, 21))
    model = Cressman()
    brain = CardiacModel(
        mesh,
        time_const,
        M_i=Mi,
        M_e=Me,
        cell_models=model,
        facet_domains=ff,
        cell_domains=None,
        ect_current=ect_current_dict
    )
    return brain


def get_solver(brain: CardiacModel) -> Any:
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "bidomain"
    ps["theta"] = 0.5
    ps["BidomainSolver"]["linear_solver_type"] = "iterative"
    ps["BidomainSolver"]["use_avg_u_constraint"] = False
    ps["CardiacODESolver"]["scheme"] = "GRL1"
    # Still unsure about units
    ps["BidomainSolver"]["Chi"] = 1.26e3      # 1/cm -- Dougherty 2015
    ps["BidomainSolver"]["Cm"] = 1.0          # muF/cm^2 -- Dougherty 2015

    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["cpp_optimize"] = True
    flags = "-O3 -ffast-math -march=native"
    df.parameters["form_compiler"]["cpp_optimize_flags"] = flags

    df.parameters.form_compiler.quadrature_degree = 1
    solver = SplittingSolver(brain, params=ps)
    return solver


def assign_initial_conditions(solver: Any) -> None:
    brain = solver.model
    model = brain.cell_models
    vs_, *_ = solver.solution_fields()
    vs_.assign(model.initial_conditions())


def get_post_processor(outpath: str, time_stamp: bool=True, home: bool=False) -> Saver:
    _outpath = Path(outpath)
    if time_stamp:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        _outpath /= timestr
    if home:
        _outpath = Path.home() / _outpath

    pp_spec = PostProcessorSpec(casedir=_outpath)
    saver = Saver(pp_spec)
    mesh = get_mesh()
    saver.store_mesh(mesh, facet_domains=None)

    field_spec = FieldSpec(save_as=("hdf5", "xdmf"), stride_timestep=1)
    saver.add_field(Field("v", field_spec))
    saver.add_field(Field("u", field_spec))

    points = np.zeros(shape=(11, 3))
    points[:, 0] = np.arange(11)/100      # To cm   range (0.0, 0.01)
    saver.add_field(PointField("point_u", field_spec, points))
    saver.add_field(PointField("point_v", field_spec, points))
    saver.add_field(PointField("point_Ca", field_spec, points))
    saver.add_field(PointField("point_K", field_spec, points))
    saver.add_field(PointField("point_Na", field_spec, points))
    return saver


def main(dt: float, T: float) -> None:
    brain = get_brain(T)
    solver = get_solver(brain)
    assign_initial_conditions(solver)
    saver = get_post_processor(outpath="out_cressman", time_stamp=True)

    theta = solver.parameters["theta"]
    for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
        norm = vs.vector().norm('l2')
        print("Timetep: {:d} -->  Norm: {:g}".format(i, norm))

        current_t = t0 + theta*(t1 - t0)
        v, u, *_ = vur.split(deepcopy=True)

        V, m, h, n, Ca, K, Na = vs.split(deepcopy=True)
        update_dict = {
            "v": v,
            "u": u,
            "point_v": v,
            "point_u": u,
            "point_Ca": Ca,
            "point_K": K,
            "point_Na": Na,
        }

        saver.update(
            brain.time,
            i,
            update_dict
        )
    saver.close()       # TODO: Add context handler?



if __name__ == "__main__":
    dt = 1e-1
    T = 10e3      # End time in [ms]
    # T = 1e2      # End time in [ms]
    main(dt, T)
