import xalbrain as xb
import numpy as np
from xalbrain.cellmodels import Wei

from ect.odesolvers import fenics_ode_solver

from ect.utilities import (
    wei_uniform_ic,
)

params = xb.SingleCellSolver.default_parameters()
model = Wei()
REFERENCE_SOLUTION = np.load("REFERENCE_SOLUTION.npy")
fire_ic = wei_uniform_ic(data=REFERENCE_SOLUTION, state="fire")

generator = fenics_ode_solver(
    model,
    dt=1e-2,
    interval=(0.0, 50.0),
    ic=fire_ic,
    params=params
)

import pandas as pd
df =  pd.DataFrame(columns=("time", "V"))

for i, (t, sol) in enumerate(generator):
    print(i)
    df.loc[i] = (t, sol[0])

df.to_pickle("ode05.pkl")
