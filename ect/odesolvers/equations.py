import time

import numpy as np


from numba import (
    jit,
    types,
    void,
)


@jit(cache=True, nopython=True, nogil=True)
def erk4_step(x, y, t0, t1):
    dt = t1 - t0

    xk1 = I(x, y, t0)
    yk1 = F(x, y, t0)

    xk2 = I(x + xk1*dt/2, y + yk1*dt/2, t0 + dt/2)
    yk2 = F(x + xk1*dt/2, y + yk1*dt/2, t0 + dt/2)

    xk3 = I(x + xk2*dt/2, y + yk2*dt/2, t0 + dt/2)
    yk3 = F(x + xk2*dt/2, y + yk2*dt/2, t0 + dt/2)

    xk4 = I(x + xk3*dt, y + yk3*dt, t0 + dt)
    yk4 = F(x + xk3*dt, y + yk3*dt, t0 + dt)

    new_x = x + dt*(xk1 + 2*xk2 + 2*xk3 + xk4)/6
    new_y = y + dt*(yk1 + 2*yk2 + 2*yk3 + yk4)/6
    return new_x, new_y


@jit(cache=True, nopython=True, nogil=True)
def solve(t_array, x_array, y_array, icx, icy):
    x_array[0] = icx
    y_array[0] = icy
    for i in range(1, t_array.size):
        new_x, new_y = erk4_step(
            x_array[i - 1],
            y_array[i - 1],
            t_array[i - 1],
            t_array[i]
        )
        x_array[i] = new_x
        y_array[i] = new_y


if __name__ == "__main__":
    from wei_model_numba import I, F
    beta0 = 7
    vol = 1.4368e-15
    volo = 1/beta0*vol

    icx = -74.30
    icy = np.array((0.0031, 0.9994, 0.0107, 4*volo, 140*vol, 144*volo, 18*vol, 130*volo, 6*vol, vol, 29.3))

    dt = 0.05
    T = 200000
    N = T/dt + 1
    t_array = np.linspace(0, T, N)
    x_array = np.zeros_like(t_array)
    y_array = np.zeros(shape=(t_array.size, len(icy)))

    start = time.clock()
    solve(t_array, x_array, y_array, icx, icy)
    print(f"time solving ic: {time.clock() - start}")
    print(x_array[-1], y_array[-1])
    np.save("V_ic.npy", x_array[::100])
    np.save("s_ic.npy", y_array[::100])
