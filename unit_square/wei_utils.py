import pickle
import numpy as np


def get_solution(filename="REFERENCE_SOLUTION.npy"):
    return np.load(filename)


def get_random_time(N, seed=42):
    data = get_solution()
    rngesus = np.random.RandomState(seed)
    indices = rngesus.random_integers(0, data.shape[0] - 1, size=N)
    return data[indices]


def get_uniform_ic(state="flat", filename="REFERENCE_SOLUTION.npy", seed=42):
    data = get_solution()
    if state == "spike":
        idx = 189800
    elif state == "flat":
        idx = 201500
    elif state == "random":
        rngesus = np.random.RandomState(seed)
        idx = rngesus.randint(0, data.shape[0] - 1)
    else:
        assert False, f"Expected state in ('spike', 'flat', 'random'), fountd {state}"

    names = ("V", "m", "h", "n", "NKo", "NKi", "NNao", "NNai", "NClo", "NCli", "vol", "O")
    return {name: val for name, val in zip(names, data[idx])}


def save_points(functions_generator, points, filename):
    # line_template = "{name}: {value}, "*len(functions)    # geberator has no length

    with open(filename, "w") as ofile:
        for functions in functions_generator:
            line = ""
            for i, f in enumerate(functions, 1):
                values = [f(p1, p2) for p1, p2 in points]
                line += "; {}: ".format(f)
                val_template = "{}, "*len(values)
                line += val_template.format(*values)
                if i % 2 == 0:
                    break
            ofile.write(line + "\n")


def pickle_points(functions_generator, points, filename):
    # Create a dict with the data  
    data_dict = {}
    data_dict["points"] = points
    for time, functions in functions_generator:
        data_dict[time] = {}
        for i, f in enumerate(functions, 1):
            values = [f(p1, p2) for p1, p2 in points]
            data_dict[time][f.__str__()] = values

    with open(filename + ".pickle", "wb") as out_handle:
        pickle.dump(data_dict, out_handle)


if __name__ == "__main__":
    ind = get_random_time(10)
    print(ind)
