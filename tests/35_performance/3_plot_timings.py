import time
import numpy as np
import matplotlib.ticker

import powerflow
import powerflow.plotting as plt


def ybusjacobi_oneiteration(Y, s, U):
    U -= (Y @ U - np.conj(s / U)) / np.diag(Y)
    return U


def ybusgaussseidel_oneiteration(Y, s, U):
    for j in range(Y.shape[0]):
        U[j] -= ((Y[j, :] @ U) - np.conj(s[j] / U[j])) / np.diag(Y)[j]
    return U


def ybusrelaxation_oneiteration(Y, s, U):
    IR = Y @ U - np.conj(s / U)
    SR = np.conj(np.conj(U) * IR)
    max_R_index = np.argmax(np.abs(SR))
    U[max_R_index] -= IR[max_R_index] / np.diag(Y)[max_R_index]
    return U


def ybusgaussseidel_plot_timings(grid, S, eps_s=1.0, max_iters=100000, acc_factor=1.0):

    for n in range(3, 100):
        grid = powerflow.mockgrids.feeder(n)
        S = powerflow.mockloads.fixed(grid, 100, 1)
        S[:, -1] = 400

        # Preparation & Intialization:
        starttime = time.time()
        grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
        Y = grid_parameters["Y_ident"]
        S = powerflow.calc_grid_parameters.integrate_slacks_for_Yident(S, grid)
        U = grid_parameters["u0"]
        deleted_nodes = grid_parameters["node_ids_to_delete"]
        S = np.delete(S, deleted_nodes, axis=1)
        diagY = np.diag(Y)
        u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
        iters_all = np.zeros(S.shape[0], dtype=np.int32)
        init_time = time.time() - starttime

        # Power Flow:
        for i in range(S.shape[0]):
            iters = 0
            s = S[i, :]

            while True:
                for j in range(Y.shape[0]):
                    U[j] -= (
                        acc_factor * ((Y[j, :] @ U) - np.conj(s[j] / U[j])) / diagY[j]
                    )
                if np.max(np.abs(U * np.conj(Y @ U) - s)) < eps_s or iters > max_iters:
                    break
                iters += 1

            u_all[i, :] = U
            iters_all[i] = iters

        solve_time = time.time() - starttime
        print(
            f"{n:2.0f} Nodes: {init_time * 1e6:5.0f} us + {int(solve_time / iters * 1e6):3.0f} us x {iters:5.0f} = {solve_time * 1e3:6.1f} ms"
        )

        # print(f'Time for init: {int(init_time * 1e6)} us')
        # print(f'Total solve time: {int(solve_time * 1e6)} us')
        # print(f'Time per Iteration: {int(solve_time / iters * 1e6)} us')

        # Output:
        runtime = time.time() - starttime
        iters = np.mean(iters_all)
        name = "YBus Jacobi"
        # print(f'{name:20s}:  {runtime:.6f} s | {iters:6.0f} | {np.min(np.abs(u_all)):.3f} V')

    return u_all, iters, runtime


grid = powerflow.mockgrids.feeder(4, R=1)
S = powerflow.mockloads.fixed(grid, 0, 1)
S[:, -1] = 400

# U, iters, runtime = plot_each_iteration(grid,S, iter_func=ybusjacobi_oneiteration, plotting=True)
# U, iters, runtime = plot_each_iteration(grid,S, iter_func=ybusgaussseidel_oneiteration, plot_filename='ybusgaussseidel_iterations', plotting=True)
U, iters, runtime = plot_timings(
    grid,
    S,
    iter_func=ybusrelaxation_oneiteration,
    plot_filename="ybusrelaxation_iterations",
    plotting=True,
)
