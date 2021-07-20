import time
starttime = time.time()
import sys
import subprocess
import numpy as np
import powerflow
import powerflow.data

method = 5  # ZBus Jacobi
nodes = 10
loads = 100

if len(sys.argv) > 1:
    method = int(sys.argv[1])
if len(sys.argv) > 2:
    nodes = int(sys.argv[2])
if len(sys.argv) > 3:
    loads = int(sys.argv[3])


grid = powerflow.data.mockgrids.feeder(nodes)
S = powerflow.data.mockloads.beta(grid, maxload=5000, numberofloads=loads, seed=4268376)

grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)

# print(f'Feeder Grid with {nodes} nodes')
# print(f'{loads} random loads with fixed seed')


# if method == 1:
#     print('Using Ybus Jacobi')
# elif method == 2:
#     print('Using Ybus Gauss-Seidel')
# elif method == 3:
#     print('Using Ybus Relaxation')
# elif method == 4:
#     print('Using Ybus Newton-Raphson')
# elif method == 5:
#     print('Using Zbus Jacobi')
# elif method == 6:
#     print('Using BFS')
# print(f'Grid setup time : {time.time() - starttime:.3}s')

while time.time() - starttime < 1:
    time.sleep(0.01)

if method == 1:
    U, iters, runtime = powerflow.ybusjacobi(grid, S, verbose=False)
elif method == 2:
    U, iters, runtime = powerflow.ybusgaussseidel(grid, S, verbose=False, acc_factor=1.0)
elif method == 3:
    U, iters, runtime = powerflow.ybusrelaxation(grid, S, verbose=False)
elif method == 4:
    U, iters, runtime = powerflow.ybusnewton(grid_parameters, S, verbose=False)
elif method == 5:
    U, iters, runtime = powerflow.zbusjacobi(grid_parameters, S, verbose=False)
elif method == 6:
    U, iters, runtime = powerflow.bfs(grid_parameters, S, verbose=False)

print(runtime)
print(int(sum(iters)))
# print(f'Finished after {runtime:.3f} s and a total of {int(sum(iters))} Iterations')


# print('Control using Ybus Newton-Raphson')
# Ucontrol, _, _ = powerflow.ybusnewton(grid, S, verbose=False)
# print(np.allclose(U, Ucontrol))
