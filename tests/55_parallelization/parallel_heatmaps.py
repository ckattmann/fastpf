import numpy as np
import mkl

mkl.set_num_threads(1)
import matplotlib.ticker as ticker

import powerflow
import powerflow.plotting as plt


def runtime_factor(func, num_nodes, num_loads):
    grid = powerflow.grids.radial(num_nodes)
    S = powerflow.loads.beta(grid, load=5000, num_loads=num_loads, seed=25345234)
    U, iters, st_runtime = func(grid, S, num_processes=1, verbose=True)
    U, iters, mt_runtime = func(grid, S, num_processes=4, verbose=True)
    return mt_runtime / st_runtime


simulations = {
    "test": {
        "alg": powerflow.ybusjacobi,
        "loads": [20, 20, 100, 1000],
        "nodes": [10, 20, 30, 40, 80],
        "filename": None,
    },
    # "ybusnewton": {
    #     "alg": powerflow.ybusnewton,
    #     "loads": [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
    #     "nodes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200],
    #     "filename": "parallel_heatmap_ybusnewton",
    # },
    # "ybusnewton2": {
    #     "alg": powerflow.ybusnewton,
    #     "loads": [20, 50, 100, 200, 500, 1000],
    #     "nodes": [10, 20, 30, 40, 50, 60, 70, 80],
    #     "filename": "parallel_heatmap_ybusnewton2B",
    # },
    # "zbusjacobi": {
    #     "alg": powerflow.zbusjacobi,
    #     "loads": [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
    #     "nodes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200],
    #     "filename": "parallel_heatmap_zbusjacobiB",
    # },
    # "zbusjacobi2": {
    #     "alg": powerflow.zbusjacobi,
    #     "loads": [20, 50, 100, 200, 500, 1000],
    #     "nodes": [10, 20, 30, 40, 50, 60, 70, 80],
    #     "filename": "parallel_heatmap_zbusjacobi2B",
    # },
    # "ybusjacobi": {
    #     "alg": powerflow.ybusjacobi,
    #     "loads": [20, 50, 100, 200, 500, 1000],
    #     "nodes": [10, 20, 30, 40, 50, 60, 70, 80],
    #     "filename": "parallel_heatmap_ybusjacobiB",
    # },
    # "ybusgaussseidel": {
    #     "alg": powerflow.ybusgaussseidel,
    #     "loads": [20, 50, 100, 200, 500, 1000],
    #     "nodes": [10, 20, 30, 40, 50, 60, 70, 80],
    #     "filename": "parallel_heatmap_ybusgaussseidelB",
    # },
}

# loads = np.logspace(1,3,5, dtype=np.int)
# nodes = np.linspace(10,100,5, dtype=np.int)

# method = "ybusnewton2"
# method = "zbusjacobi2"
method = "test"

loads = simulations[method]["loads"]
nodes = simulations[method]["nodes"]
func = simulations[method]["alg"]

all_factors = np.zeros((len(loads), len(nodes)))
for i, num_loads in enumerate(loads):
    for j, num_nodes in enumerate(nodes):
        print(f"{num_nodes} Nodes, {num_loads} Loads: ")
        all_factors[i, j] = runtime_factor(func, num_nodes, num_loads)

print("Plotting...")
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={"width_ratios": (15, 1)})
plt.setsize(fig, 1)

image = ax.matshow(all_factors, vmin=0, vmax=2, cmap="RdYlGn_r", origin="lower")

ax.set_ylabel("Number of Loads")
ax.set_yticks(range(0, len(loads)))
ax.set_yticklabels(loads)

ax.set_xlabel("Number of Nodes")
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
ax.set_xticks(range(0, len(nodes)))
ax.set_xticklabels(nodes)
if len(nodes) > 8:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

cbar = fig.colorbar(image, cax=cbar_ax)
cbar_ax.yaxis.set_label_position("right")
cbar.ax.set_ylabel("Runtime Factor / -", rotation=90, labelpad=10)

fig.tight_layout()
if simulations[method]["filename"]:
    plt.save(fig, simulations[method]["filename"])

# plt.show()
