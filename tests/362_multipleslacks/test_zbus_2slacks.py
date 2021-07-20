import numpy as np
import powerflow
import powerflow.plotting as plt

# 1 Slack
grid = powerflow.grids.feeder(10, numslacks=1, slack_voltages=[400,390])
S = powerflow.loads.random(grid, load=4000, num_loads=100)
U, iters, runtime = powerflow.zbusjacobi(grid,S)

# 2 Slacks
grid = powerflow.grids.feeder(10, numslacks=2, slack_voltages=[400,390])
S = powerflow.loads.random(grid, load=4000, num_loads=100)
U, iters, runtime = powerflow.zbusjacobi(grid,S)

# >3 Slacks
slack_voltages = [110000,112000, 115000, 110000, 100000]
grid = powerflow.grids.meshed(110, voltage_level='h', numslacks=len(slack_voltages), slack_voltages=slack_voltages, num_crosses=0)
S = powerflow.loads.random(grid, load=1e6, num_loads=100)
U, iters, runtime = powerflow.zbusjacobi(grid,S)
U2, iters, runtime = powerflow.ybusnewton(grid,S)

print(np.max(U2-U))
print(np.mean(U2-U))


plt.plotgraph(grid, shape='spring')
plt.show()

plt.plot(np.abs(U).T)
plt.grid(True)
plt.show()
