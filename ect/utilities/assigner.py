import numpy as np
import xalbrain as xb


def assign_ic(func):
    mixed_func_space = func.function_space()

    functions = func.split(deepcopy=True)
    ic = get_random_time(functions[0].vector().size())
    V = xb.FunctionSpace(mixed_func_space.mesh(), "CG", 1)

    for i, f in enumerate(functions):
        ic_func = xb.Function(V)
        ic_func.vector()[:] = np.array(ic[:, i])

        assigner = xb.FunctionAssigner(mixed_func_space.sub(i), V)
        assigner.assign(func.sub(i), ic_func)
