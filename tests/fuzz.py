import numpy as np
import fastpf

grid = fastpf.testgrids.radial(30)
S = fastpf.testloads.random_uniform(grid, max_load_W=1000, number_of_scenarios=10000)

U_yjacobi, all_converged, iters, runtime = fastpf.ybusjacobi(grid, S)
U_yjacobi, all_converged, iters, runtime = fastpf.ybusjacobi(grid, S)
U_ygaussseidel, all_converged, iters, runtime = fastpf.ybusgaussseidel(grid, S)
U_newton, all_converged, iters, runtime = fastpf.ybusnewton(grid, S)
U_zbus, all_converged, iters, runtime = fastpf.zbusjacobi(grid, S)

print(np.allclose(U_newton, U_zbus, U_yjacobi, U_ygaussseidel))

fastpf.powerflow.compare_methods(grid, S)
