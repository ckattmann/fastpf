# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import mkl

# mkl.set_num_threads(1)


import time
import functools

from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np

from . import powerflow_methods_cc
from . import powerflow_methods
from . import calc_grid_parameters

VERBOSE = 1
eps_s = 1


def print_result(name, runtime, mean_iters, min_U):
    runtime_ms = runtime * 1000
    print(f"{name:20s}:  {runtime:.6f} s | {mean_iters:6.0f} | {min_U:.3f} V")
    # print(f'{name:25s} & {mean_iters:6.0f} & {runtime_ms:8.3f} ms \\\\')


def parallelize(func, S, n):
    mkl.set_num_threads(1)
    borders = np.linspace(0, S.shape[0], n + 1, dtype=np.int)
    S_chunks = [S[borders[i] : borders[i + 1], :] for i in range(n)]
    # print(' | '.join([str(s.shape[0]) for s in S_chunks]))
    with Pool(n) as pool:
        all_results = pool.map(func, S_chunks)
    U, iters = (result_set for result_set in zip(*all_results))
    U = np.vstack(U)
    iters = np.hstack(iters)
    mkl.set_num_threads(mkl.get_max_threads())
    return U, iters


def _parse_grid(grid):
    if "edges" in grid:
        grid_parameters = calc_grid_parameters.prepdata(grid)
    elif "Y" in grid:
        grid_parameters = grid
    return grid_parameters


def ybusjacobi(grid, S, eps_s=1, max_iters=20000, num_processes=1, verbose=VERBOSE):
    starttime = time.time()
    grid_parameters = _parse_grid(grid)
    Y = grid_parameters["Y_ident"]
    S = calc_grid_parameters.integrate_slacks_for_Yident(S, grid)
    U0 = grid_parameters["u0"]
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    S = np.delete(S, deleted_nodes, axis=1)
    if num_processes == 1:
        U, iters = powerflow_methods_cc.ybusjacobi(Y, U0, S, eps_s, max_iters)
    else:
        func = lambda S: powerflow_methods_cc.ybusjacobi(Y, U0, S, eps_s, max_iters)
        U, iters = parallelize(func, S, num_processes)
    runtime = time.time() - starttime
    if verbose:
        print_result("Ybus Jacobi", runtime, np.mean(iters), np.min(np.abs(U)))
    return U, iters, runtime


def ybusgaussseidel(
    grid,
    S,
    eps_s=1,
    max_iters=10000,
    acc_factor=1.6,
    num_processes=1,
    lines_to_remove=[],
    verbose=VERBOSE,
):
    starttime = time.time()
    grid_parameters = calc_grid_parameters.calc_grid_parameters(
        grid, S, lines_to_remove=lines_to_remove
    )
    Y = grid_parameters["Y_ident"]
    U0 = grid_parameters["u0"]

    U0 = calc_grid_parameters.put_slack_voltages_into_u0(grid, 118000)

    S = calc_grid_parameters.integrate_slacks_for_Yident(S, grid)
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    S = np.delete(S, deleted_nodes, axis=1)
    if num_processes == 1:
        U, iters = powerflow_methods_cc.ybusgaussseidel(
            Y, U0, S, acc_factor, eps_s, max_iters
        )
    else:
        func = lambda S: powerflow_methods_cc.ybusgaussseidel(
            Y, U0, S, acc_factor, eps_s, max_iters
        )
        U, iters = parallelize(func, S, num_processes)
    runtime = time.time() - starttime
    if verbose:
        print_result("Ybus Gauss-Seidel", runtime, np.mean(iters), np.min(np.abs(U)))
    return U, iters, runtime


def ybusrelaxation(grid, S, eps_s=1, max_iters=100000, verbose=VERBOSE):
    starttime = time.time()
    grid_parameters = _parse_grid(grid)
    Y = grid_parameters["Y_ident"]
    U0 = grid_parameters["u0"]
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    S = np.delete(S, deleted_nodes, axis=1)
    U, iters = powerflow_methods_cc.ybusrelaxation(Y, U0, S, eps_s, max_iters)
    runtime = time.time() - starttime
    if verbose:
        print_result("Ybus Relaxation", runtime, np.mean(iters), np.min(np.abs(U)))
    return U, iters, runtime


def ybusnewton(
    grid,
    S,
    eps_s=1,
    max_iters=100,
    num_processes=1,
    sparse=False,
    lines_to_remove=[],
    verbose=VERBOSE,
):
    starttime = time.time()
    grid_parameters = calc_grid_parameters.calc_grid_parameters(
        grid,
        S,
        reorder_slack_nodes=True,
        lines_to_remove=lines_to_remove,
        verbose=verbose,
    )
    Y = grid_parameters["Y"]
    Yred = grid_parameters["Yred"]
    U0 = grid_parameters["u0"]
    num_slacks = grid_parameters["numslacks"]
    if verbose >= 2:
        print(f"Setup Time: {time.time() - starttime:8.3f} s")

    # if grid_parameters['node_swaps']:
    #     node_swaps = grid_parameters['node_swaps'].copy()
    #     print(node_swaps)
    #     while node_swaps:
    #         id1, id2 = node_swaps.pop()
    #         print(f'Swapping nodes {id1:3d},{id2:3d}')
    #         S[:,[id2, id1]] = S[:,[id1, id2]]

    if num_slacks > 1:
        additional_slack_voltages = grid_parameters["additional_slack_voltages"]
        for i, u_slack in enumerate(additional_slack_voltages, 1):
            U0[i] = u_slack
        if num_processes == 1:
            U, iters = powerflow_methods_cc.ybusnewton_mslacks(
                Y, S, U0, num_slacks, eps_s, max_iters
            )
        else:
            func = lambda S: powerflow_methods_cc.ybusnewton_mslacks(
                Y, S, U0, num_slacks, eps_s, max_iters
            )
            U, iters = parallelize(func, S, num_processes)
    else:
        if sparse:
            slack_index = grid_parameters["slack_index"]
            deleted_nodes = grid_parameters["node_ids_to_delete"]
            S = np.delete(S, deleted_nodes, axis=1)
            if num_processes == 1:
                U, iters = powerflow_methods.ybusnewton_sparse(
                    Y, Yred, U0, S, slack_index, eps_s, max_iters
                )
            else:
                par_func = lambda S: powerflow_methods.ybusnewton_sparse(
                    Y, Yred, U0, S, slack_index, eps_s, max_iters
                )
                U, iters = parallelize(par_func, S, num_processes)
        else:
            if num_processes == 1:
                # U, iters = powerflow_methods_cc.ybusnewton(Y, Yred, U0, S, slack_index, eps_s, max_iters)
                U, iters = powerflow_methods_cc.ybusnewton(Y, S, U0, eps_s, max_iters)
            else:
                func = lambda S: powerflow_methods_cc.ybusnewton(
                    Y, S, U0, eps_s, max_iters
                )

                U, iters = parallelize(func, S, num_processes)

    if grid_parameters["node_swaps"]:
        node_swaps = grid_parameters["node_swaps"].copy()
        while node_swaps:
            id1, id2 = node_swaps.pop()
            U[:, [id2, id1]] = U[:, [id1, id2]]

    runtime = time.time() - starttime
    if verbose >= 1:
        print_result("Ybus Newton", runtime, np.mean(iters), np.min(np.abs(U)))
    return U, iters, runtime


def zbusjacobi(
    grid,
    S,
    eps_s=1,
    max_iters=100,
    num_processes=1,
    sparse=False,
    lines_to_remove=[],
    verbose=VERBOSE,
):
    starttime = time.time()
    # grid_parameters = _parse_grid(grid)
    grid_parameters = calc_grid_parameters.calc_grid_parameters(
        grid,
        S,
        collapse_identical_slacks=False,
        reorder_slack_nodes=True,
        lines_to_remove=lines_to_remove,
        verbose=False,
    )
    Zred = np.linalg.inv(grid_parameters["Yred"])
    U0 = grid_parameters["u0"]
    slack_voltage = U0[0]
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    slack_index = grid_parameters["slack_index"]
    additional_slack_ids = grid_parameters["additional_slack_ids"]
    S[:, additional_slack_ids] = 0
    # print(time.time() - starttime)

    # if grid_parameters['node_swaps']:
    #     node_swaps = grid_parameters['node_swaps'].copy()
    #     print(node_swaps)
    #     while node_swaps:
    #         id1, id2 = node_swaps.pop()
    #         S[:,[id2, id1]] = S[:,[id1, id2]]

    # print(deleted_nodes, slack_index)

    S = np.delete(S, deleted_nodes + [slack_index], axis=1)

    ## 2 Slacks:
    if grid_parameters["numslacks"] == 2:
        i_slack = (
            additional_slack_ids[0] - 1
        )  # -1 because the slack index 0 is deleted in Zred and S
        U_slack2 = grid_parameters["additional_slack_voltages"][0]
        eps_u_slack = 0.001
        U, iters = powerflow_methods_cc.zbusjacobi_2slacks(
            Zred, S, U0, slack_voltage, i_slack, U_slack2, eps_s, eps_u_slack, max_iters
        )

    ## n Slacks:
    elif grid_parameters["numslacks"] > 2:
        m = len(grid_parameters["additional_slack_voltages"])
        U_additional_slacks = grid_parameters["additional_slack_voltages"]
        eps_u_slack = 0.001
        if num_processes == 1:
            # print('zbus, nslacks:')
            U, iters = powerflow_methods_cc.zbusjacobi_mslacks(
                Zred,
                S,
                U0,
                slack_voltage,
                m,
                U_additional_slacks,
                eps_s,
                eps_u_slack,
                max_iters,
            )
        else:
            par_func = lambda S: powerflow_methods_cc.zbusjacobi_mslacks(
                Zred,
                S,
                U0,
                slack_voltage,
                m,
                U_additional_slacks,
                eps_s,
                eps_u_slack,
                max_iters,
            )
            U, iters = parallelize(par_func, S, num_processes)
    else:
        # One slack:
        if sparse:
            if num_processes == 1:
                U, iters = powerflow_methods.zbusjacobi_sparse(
                    Zred, S, slack_voltage, eps_s, max_iters
                )
            else:
                func = lambda S: powerflow_methods.zbusjacobi_sparse(
                    Zred, S, slack_voltage, eps_s, max_iters
                )
                U, iters = parallelize(func, S, num_processes)
        else:
            if num_processes == 1:
                U, iters = powerflow_methods_cc.zbusjacobi(
                    Zred, S, slack_voltage, eps_s, max_iters
                )
            else:
                func = lambda S: powerflow_methods_cc.zbusjacobi(
                    Zred, S, slack_voltage, eps_s, max_iters
                )
                U, iters = parallelize(func, S, num_processes)

    # If slack nodes have been swapped to the top, swap them back so that the original order is restored:
    if grid_parameters["node_swaps"]:
        node_swaps = grid_parameters["node_swaps"].copy()
        while node_swaps:
            id1, id2 = node_swaps.pop()
            U[:, [id2, id1]] = U[:, [id1, id2]]

    runtime = time.time() - starttime

    if verbose:
        print_result("Zbus Jacobi", runtime, np.mean(iters), np.min(np.abs(U)))

    return U, iters, runtime


def bfs(grid, S, eps_s=1, max_iters=1000, verbose=VERBOSE):
    starttime = time.time()
    grid_parameters = _parse_grid(grid)
    Zline = grid_parameters["Zline"]
    if Zline is None:
        name = "& Backward/Forward Sweep"
        print(f"{name:25s} & - & - \\\\")
        # print('Error: BFS is impossible because grid is not a feeder')
        return 0, 0, 0
    slack_voltage = grid_parameters["u0"][0]
    U, iters = powerflow_methods_cc.bfs(Zline, S, slack_voltage, eps_s, max_iters)
    runtime = time.time() - starttime
    if verbose:
        print_result(
            "Backward/Forward Sweep", runtime, np.mean(iters), np.min(np.abs(U))
        )
    return U, iters, runtime
