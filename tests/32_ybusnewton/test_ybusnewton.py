import powerflow
import powerflow.plotting as plt


grid = powerflow.grids.feeder(30)
S = powerflow.loads.beta(grid, maxload=10000, numberofloads=100000)

U, iters, runtime = powerflow.ybusnewton(grid,S)

# plt.plot(U.T)
# plt.grid(True)
# plt.show()
