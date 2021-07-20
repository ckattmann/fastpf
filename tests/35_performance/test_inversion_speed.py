import time
import numpy as np
import powerflow.plotting as plt

n = 10000

nodes = []
times = []
for n in [10, 20, 50, 100, 200, 500, 1000, 2000]:
    nodes.append(n)
    Y = np.random.random((n, n))
    inv_times = []
    for i in range(10):
        starttime = time.time()
        Z = np.linalg.inv(Y)
        inv_times = time.time() - starttime
    inv_time = np.mean(inv_times)
    times.append(inv_time)
    print(inv_time)

fig, ax = plt.subplots()
print(nodes, times)
ax.semilogx(nodes, times, "o-")
ax.grid(True)
plt.setsize(fig)
plt.show()
