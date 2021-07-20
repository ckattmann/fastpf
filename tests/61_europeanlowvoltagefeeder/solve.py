import sys
import time
import json
import numpy as np
import powerflow
import matplotlib.pyplot as plt

import grid_reduction

# Parse number of loads from command line
if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 14

S = np.load("europeanLVFeeder.npy")[:n, :].astype(np.complex128)
plt.plot(S)
plt.show()

with open("european_lv_feeder.json", "r") as f:
    grid = json.load(f)


# No reduction:
# =============

U, iters, runtime = powerflow.powerflow.ybusnewton(grid, S, sparse=False)
# U, iters, runtime = powerflow.powerflow.zbusjacobi(grid, S, sparse=False)
# U, iters, runtime = powerflow.powerflow.ybusgaussseidel(grid, S, acc_factor=1.0, max_iters=100000)
# U, iters, runtime = powerflow.powerflow.ybusjacobi(grid, S)

plt.plot(np.abs(U.T))
plt.show()

sys.exit(0)


# Grid Reduction
# ==============

# starttime = time.time()
# reduced_grid2, nodes_deleted2 = grid_reduction.reduce_unnecessary_nodes(grid)
# print(time.time() - starttime)
# starttime = time.time()
reduced_grid, nodes_deleted = grid_reduction.reduce_grid(grid, verbose=True)
# reduced_grid = grid_reduction.normalize_node_ids(reduced_grid)
# print(time.time() - starttime)
# print(len(reduced_grid['nodes']), len(reduced_grid['edges']), len(nodes_deleted))
# print(len(reduced_grid2['nodes']), len(reduced_grid2['edges']), len(nodes_deleted2))

# with open('european_lv_feeder_reduced.json','w') as f:
#     json.dump(reduced_grid, f)

S = np.load("europeanLVFeeder.npy")[:n, :].astype(np.complex128)
# print(S.shape)
# S = np.delete(S, nodes_deleted, axis=1)
# print(S.shape)

# U, iters, runtime = powerflow.powerflow.ybusnewton(reduced_grid, S)
U, iters, runtime = powerflow.powerflow.zbusjacobi(reduced_grid, S)
# U, iters, runtime = powerflow.powerflow.ybusgaussseidel(reduced_grid, S)
# U, iters, runtime = powerflow.powerflow.ybusjacobi(reduced_grid, S)


# U, iters, runtime = powerflow.powerflow.ybusnewton(grid, S, sparse=True)
U, iters, runtime = powerflow.powerflow.zbusjacobi(reduced_grid, S, sparse=True)
