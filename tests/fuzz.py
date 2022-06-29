import numpy as np
import fastpf


def fuzz_test(number_of_nodes, max_load_W):
    grid = fastpf.testgrids.radial(number_of_nodes)
    S = fastpf.testloads.random_uniform(
        grid, max_load_W=max_load_W, number_of_scenarios=100
    )

    U_yjacobi, all_converged, iters, runtime = fastpf.ybusjacobi(grid, S)
    U_yjacobi, all_converged, iters, runtime = fastpf.ybusjacobi(grid, S)
    U_ygaussseidel, all_converged, iters, runtime = fastpf.ybusgaussseidel(grid, S)
    U_newton, all_converged, iters, runtime = fastpf.ybusnewton(grid, S)
    U_zbus, all_converged, iters, runtime = fastpf.zbusjacobi(grid, S)

    assert np.allclose(
        U_newton, U_zbus, U_yjacobi, U_ygaussseidel
    ), f"Fuzz Test failed with"

    # fastpf.powerflow.compare_methods(grid, S)


for i in range(20):
    number_of_nodes = np.random.randint(2, 100)
    max_load_W = np.random.randint(2, 40000)
    fuzz_test(number_of_nodes, max_load_W)
