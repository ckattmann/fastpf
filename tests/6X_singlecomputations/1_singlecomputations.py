import time
import json
import numpy as np
import powerflow.powerflow as pf
import powerflow.data.mockgrids
import powerflow.data.mockloads
import powerflow.calc_grid_parameters

mockgrids = powerflow.data.mockgrids

gridfunction = mockgrids.feeder

all_results = {}

for gridfunction in [
    mockgrids.feeder,
    mockgrids.feeder,
    mockgrids.radial,
    mockgrids.meshed,
]:

    print(gridfunction.__name__)
    starttime = time.time()
    grid = gridfunction(40)
    S = powerflow.data.mockloads.fixed(grid, load=5000)
    # print(time.time() - starttime)

    # all_results[gridfunction.__name__] = {}
    results = {}

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    U, iters, runtime = pf.ybusjacobi(grid_parameters, S)
    results["ybusjacobi"] = {
        "minU": float(np.min(np.abs(U))),
        "iters": int(np.mean(iters)),
        "runtime": float(runtime),
    }

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    U, iters, runtime = pf.ybusgaussseidel(grid_parameters, S)
    results["ybusgaussseidel"] = {
        "minU": float(np.min(np.abs(U))),
        "iters": int(np.mean(iters)),
        "runtime": float(runtime),
    }

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    U, iters, runtime = pf.ybusrelaxation(grid_parameters, S)
    results["ybusrelaxation"] = {
        "minU": float(np.min(np.abs(U))),
        "iters": int(np.mean(iters)),
        "runtime": float(runtime),
    }

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    U, iters, runtime = pf.ybusnewton(grid_parameters, S)
    results["ybusnewton"] = {
        "minU": float(np.min(np.abs(U))),
        "iters": int(np.mean(iters)),
        "runtime": float(runtime),
    }

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    U, iters, runtime = pf.zbusjacobi(grid_parameters, S)
    results["zbusjacobi"] = {
        "minU": float(np.min(np.abs(U))),
        "iters": int(np.mean(iters)),
        "runtime": float(runtime),
    }

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    U, iters, runtime = pf.bfs(grid_parameters, S)
    results["bfs"] = {
        "minU": float(np.min(np.abs(U))),
        "iters": int(np.mean(iters)),
        "runtime": float(runtime),
    }

    # [print(res['minU']) for res in results.values()]
    all_results[gridfunction.__name__] = results

    # import pprint
    # pprint.pprint(results)

with open("results1.json", "w") as f:
    json.dump(all_results, f)
