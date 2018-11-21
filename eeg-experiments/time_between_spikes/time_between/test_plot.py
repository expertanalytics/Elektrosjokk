import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

distance = np.load("distance.npy")
time = np.load("time.npy")
print("distance: ", distance.min(), distance.mean(), distance.max())
print("time: ", time.min(), time.mean(), time.max())

ax_time_distance = sns.boxplot(time/distance, whis=1)
fig_time_distance = ax_time_distance.get_figure()
ax_time_distance.set_title("Time to next electrode spike / Distance to electrode")
ax_time_distance.set_xlabel("time spike/distance to electrode")
fig_time_distance.savefig("time_distance_boxplot.png")

# ax_time = sns.boxplot(time, whis=1)
# fig_time = ax_time.get_figure()
# ax_time.set_title("Time to next electrode spike / Distance to electrode")
# ax_time.set_xlabel("time spike/distance to electrode")
# fig_time.savefig("time_boxplot.png")

# ax_distance = sns.boxplot(distance, whis=1)
# fig_distance = ax_distance.get_figure()
# ax_distance.set_title("Time to next electrode spike / Distance to electrode")
# ax_distance.set_xlabel("time spike/distance to electrode")
# fig_distance.savefig("distance_boxplot.png")

# fig, ax = plt.subplots(1)
# ax.boxplot(data)
# print(data.min(), data.mean(), data.max())

# fig.savefig("foo.png")
