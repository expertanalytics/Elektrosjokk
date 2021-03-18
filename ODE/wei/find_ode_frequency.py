import numpy as np
import matplotlib.pyplot as plt


ode_data = np.load("ode_values.npy")

N = 70000*100
time_array = ode_data[N: N + 10000, 0]/100      # Scale to ms
V_array = ode_data[N: N + 10000, 1]

plt.plot(time_array, V_array)
plt.savefig("V_frequency.png")
