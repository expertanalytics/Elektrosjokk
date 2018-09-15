import dolfin as df
import numpy as np
import pandas as pd

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

from xalbrain.cellmodels import Wei

from postfields import (
    Field,
    PointField,
)

from postspec import (
    FieldSpec,
    PostProcessorSpec,
)

from post import Saver

from pathlib import Path
import time


def assign_ic(func, data):
    mixed_func_space = func.function_space()

    functions = func.split(deepcopy=True)
    V = df.FunctionSpace(mixed_func_space.mesh(), "CG", 1)
    ic_indices = np.random.randint(
        0,
        data.shape[0],
        size=functions[0].vector().local_size()
    )
    _data = data[ic_indices]

    for i, f in enumerate(functions):
        ic_func = df.Function(V)
        ic_func.vector()[:] = np.array(_data[:, i])

        assigner = df.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(func.sub(i), ic_func)


time_const = df.Constant(0)
dt = 1e-2
T = 1e3      # End time in [ms]
mesh = df.UnitSquareMesh(10, 10)       # 1cm time 1cm

Mi = df.Constant(2)    # TODO: look these up   in mS/cm
Me = df.Constant(1)

ic_data = np.load(Path.home() / "Documents/ECT-data/initial_conditions/REFERENCE_SOLUTION.npy")

model = Wei()
brain = CardiacModel(
    mesh,
    time_const,
    M_i=Mi,
    M_e=Me,
    cell_models=model,
    facet_domains=None,
    cell_domains=None,
    ect_current=None
)

ps = SplittingSolver.default_parameters()
ps["pde_solver"] = "bidomain"
ps["theta"] = 0.5
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["use_avg_u_constraint"] = False

# Still unsure about units
ps["BidomainSolver"]["Chi"] = 1.26e3      # 1/cm -- Dougherty 2015
ps["BidomainSolver"]["Cm"] = 1.0          # muF/cm^2 -- Dougherty 2015

df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
df.parameters["form_compiler"]["cpp_optimize_flags"] = flags

df.parameters.form_compiler.quadrature_degree = 1
solver = SplittingSolver(brain, params=ps)

vs_, *_ = solver.solution_fields()
assign_ic(vs_, ic_data)

assert np.unique(vs_.vector().array()).size > 12

field_spec = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=1)
outpath =  Path.home() / "out/bergen_casedir"
pp_spec = PostProcessorSpec(casedir=outpath)

saver = Saver(pp_spec)
saver.store_mesh(mesh, facet_domains=None)
saver.add_field(Field("v", field_spec))
saver.add_field(Field("u", field_spec))

theta = ps["theta"]

for i, ((t0, t1), (vs_, vs, vur)) in enumerate(solver.solve((0, T), dt)):
    norm = vs.vector().norm('l2')
    print("Solving ({t0}, {t1}) -----> {norm}\n".format(t0=t0, t1=t1, norm=norm))
    current_t = t0 + theta*(t1 - t0)
    v, u, *_ = vur.split(deepcopy=True)

    update_dict = {"v": v, "u": u}

    saver.update(
        time_const,
        i,
        update_dict
    )
saver.close()
