import numpy as np

def get_solution(filename="wei_solution.npy"):
    return np.load(filename)


def get_random_time(N, seed=42):
    data = get_solution()
    indices = np.random.random_integers(0, data.shape[0], size=N)
    return data[indices]


if __name__ == "__main__":
    ind = get_random_time(10)
    print(ind)
