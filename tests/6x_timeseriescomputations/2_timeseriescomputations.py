import time
import json
import numpy as np
import powerflow.powerflow as pf
import powerflow.data.mockgrids
import powerflow.data.mockloads
import powerflow.calc_grid_parameters

mockgrids = powerflow.data.mockgrids


# print(gridfunction.__name__)

# grid = mockgrids.feeder(10)
grid = mockgrids.ngrot()

# S = powerflow.data.mockloads.beta(grid, maxload=20000, numberofloads=100)
# S = powerflow.data.mockloads.fixed(grid, load=1000, numberofloads=1)
S = powerflow.data.mockloads.ngrot(grid, n=10)
print(S.shape)

results = {}

# grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
# U, iters, runtime = pf.ybusjacobi(grid_parameters, S)
# results['ybusjacobi'] = {'minU':float(np.min(np.abs(U))), 'iters':int(np.mean(iters)), 'runtime':float(runtime)}

grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
U, iters, runtime = pf.ybusgaussseidel(grid_parameters, S)
results['ybusgaussseidel'] = {'minU':float(np.min(np.abs(U))), 'iters':int(np.mean(iters)), 'runtime':float(runtime)}

# grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
# U, iters, runtime = pf.ybusrelaxation(grid_parameters, S)
# results['ybusrelaxation'] = {'minU':float(np.min(np.abs(U))), 'iters':int(np.mean(iters)), 'runtime':float(runtime)}

grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
U, iters, runtime = pf.ybusnewton(grid_parameters, S)
results['ybusnewton'] = {'minU':float(np.min(np.abs(U))), 'iters':int(np.mean(iters)), 'runtime':float(runtime)}

grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
U, iters, runtime = pf.zbusjacobi(grid_parameters, S)
results['zbusjacobi'] = {'minU':float(np.min(np.abs(U))), 'iters':int(np.mean(iters)), 'runtime':float(runtime)}

# grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
# U, iters, runtime = pf.bfs(grid_parameters, S)
# results['bfs'] = {'minU':float(np.min(np.abs(U))), 'iters':int(np.mean(iters)), 'runtime':float(runtime)}

import pprint
pprint.pprint(results)

# with open('results1.json','w') as f:
#     json.dump(results,f)

