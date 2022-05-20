import time
import numpy as np

# threadpoolctl allows control over the number of threads used by numpy,
# regardless of the underlying blas:
from threadpoolctl import threadpool_limits

# pathos allows multiprocessing with numpy arrays as parameters:
from pathos.multiprocessing import ProcessingPool

from . import powerflow_methods_cc as pf_cc
from . import process_grid
from . import integrate_slacks_for_Yident
from .log import logger
from .log import scale_time
from .log import set_loglevel

# This line replaces the compiled PF version with the Python versions, for development:
# from . import powerflow_methods as pf_cc


def _log_result(name, runtime_s, mean_iters, min_U, all_converged=True):
    runtime_string = scale_time(runtime_s)
    if all_converged:
        logger.success(
            f"{name:20s} | {runtime_string:>10} | {mean_iters:6.0f} | {min_U:.3f} V"
        )
    else:
        logger.warning(
            f"{name:20s} | {runtime_string:>10} | {mean_iters:6.0f} | {min_U:.3f} V - contains nonconverged results"
        )


def _parallelize(func, S, num_processes=1):
    # Shortcut for the case num_processes=1
    if num_processes == 1:
        U, iters = func(S)
        return U, iters

    if num_processes > 1:
        # Break up S into chunks:
        borders = np.linspace(0, S.shape[0], num_processes + 1, dtype=int)
        S_chunks = [S[borders[i] : borders[i + 1], :] for i in range(num_processes)]

        # Then process in parallel:
        with threadpool_limits(limits=1, user_api="blas"):
            with ProcessingPool(num_processes) as pool:
                all_results = pool.map(func, S_chunks)

        # And re-combine the results:
        U, iters = (result_set for result_set in zip(*all_results))
        U = np.vstack(U)
        iters = np.hstack(iters)

        return U, iters

    else:
        raise ValueError("num_processes must be > 1")


def ybusjacobi(grid, S, eps_s=1, max_iters=20000, num_processes=1):
    """Execute a power flow using the YBus Jacobi method.

    YBUS Jacobi has fast startup, is very memory-efficient and has very fast individual iterations, but potentially requires many iterations (>10000) to find a solution.

    ### Parameters:

    `grid`: *dict*, **required**
    > The grid dict (see ´fastpf.process_grid´) for which the power flow is executed

    `S`: *numpy array, complex 128*, **required**
    > The loads for which the power flow is executed, structured in a numpy array with *number_of_scenarios* rows (axis 0), and *number_of_nodes* columns (axis 1)

    `eps_s`: *int* or *float*, *Default: 1.0*
    > The accurracy at which an iterated result is accepted as final, in *VA*

    `max_iters`: *int*, *Default: 20000*
    > The number of iterations after which the iterations are interrupted. The intermediate, non-converged result is still stored in U, and the number of iterations stored in `iters` is max_iters + 1, so non-converged results can be checked with any(iters > max_iters)

    `num_processes`: *int*, *Default: 1*
    > Number of processes used for the computation. If >1, then `S` is chunked and recombined after the computation.

    ### Returns:
    `U`: *numpy array, complex128*
    > Resulting voltages

    `all_converged`: *Bool*
    > Is `False` if there are any non-converged results in U

    `iters`: *numpy array, int64*
    > Array with the iteration counts voltages

    `runtime_s`: *float*
    > Total runtime of the method in seconds
    """

    starttime = time.perf_counter()

    # Process grid and get relevant vars:
    grid_parameters = process_grid(grid, S)
    Y = grid_parameters["Y_ident"]
    U0 = grid_parameters["u0"]
    deleted_nodes = grid_parameters["node_ids_to_delete"]

    # Modify S to support slacks and deleted nodes:
    S = integrate_slacks_for_Yident(S, grid)
    S = np.delete(S, deleted_nodes, axis=1)

    # Perform the actual power flow in the compiled function:
    func = lambda S: pf_cc.ybusjacobi(Y, U0, S, eps_s, max_iters)
    U, iters = _parallelize(func, S, num_processes)

    # Check if all scenarios have converged:
    all_converged = all(iters <= max_iters)

    runtime_s = time.perf_counter() - starttime

    _log_result(
        "Ybus Jacobi", runtime_s, np.mean(iters), np.min(np.abs(U)), all_converged
    )

    return U, all_converged, iters, runtime_s


def ybusgaussseidel(
    grid,
    S,
    eps_s=1,
    max_iters=10000,
    acc_factor=1.6,
    num_processes=1,
):

    starttime = time.perf_counter()

    grid_parameters = process_grid(grid, S)
    Y = grid_parameters["Y_ident"]
    U0 = grid_parameters["u0"]

    S = integrate_slacks_for_Yident(S, grid)

    func = lambda S: pf_cc.ybusgaussseidel(Y, U0, S, acc_factor, eps_s, max_iters)
    U, iters = _parallelize(func, S, num_processes)

    all_converged = all(iters <= max_iters)

    runtime = time.perf_counter() - starttime

    _log_result(
        "Ybus Gauss Seidel", runtime, np.mean(iters), np.min(np.abs(U)), all_converged
    )

    return U, all_converged, iters, runtime


def _ybusrelaxation(grid, S, eps_s=1, max_iters=100000):
    starttime = time.perf_counter()
    grid_parameters = _parse_grid(grid)
    Y = grid_parameters["Y_ident"]
    U0 = grid_parameters["u0"]
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    S = np.delete(S, deleted_nodes, axis=1)
    U, iters = pf_cc.ybusrelaxation(Y, U0, S, eps_s, max_iters)
    runtime = time.perf_counter() - starttime
    if verbose:
        _log_result("Ybus Relaxation", runtime, np.mean(iters), np.min(np.abs(U)))
    return U, iters, runtime


def ybusnewton(
    grid,
    S,
    eps_s=1,
    max_iters=100,
    num_processes=1,
    sparse=False,
):
    starttime = time.perf_counter()

    grid_parameters = process_grid(
        grid,
        S,
        reorder_slack_nodes=True,
    )
    Y = grid_parameters["Y"]
    Yred = grid_parameters["Yred"]
    U0 = grid_parameters["u0"]
    num_slacks = grid_parameters["numslacks"]

    logger.debug(f"Ybus Newton Setup Time: {time.perf_counter() - starttime:8.3f} s")
    if num_slacks > 1:
        additional_slack_voltages = grid_parameters["additional_slack_voltages"]
        m = len(additional_slack_voltages)
        for i, u_slack in enumerate(additional_slack_voltages, 1):
            U0[i] = u_slack
        func = lambda S: pf_cc.ybusnewton_mslacks(Y, S, U0, m, eps_s, max_iters)
    else:
        if sparse:
            slack_index = grid_parameters["slack_index"]
            deleted_nodes = grid_parameters["node_ids_to_delete"]
            S = np.delete(S, deleted_nodes, axis=1)

            func = lambda S: powerflow_methods.ybusnewton_sparse(
                Y, Yred, U0, S, slack_index, eps_s, max_iters
            )
        else:
            func = lambda S: pf_cc.ybusnewton(Y, S, U0, eps_s, max_iters)

    U, iters = _parallelize(func, S, num_processes)

    if grid_parameters["node_swaps"]:
        node_swaps = grid_parameters["node_swaps"].copy()
        while node_swaps:
            id1, id2 = node_swaps.pop()
            U[:, [id2, id1]] = U[:, [id1, id2]]

    all_converged = all(iters <= max_iters)

    runtime = time.perf_counter() - starttime

    _log_result(
        "Ybus Newton", runtime, np.mean(iters), np.min(np.abs(U)), all_converged
    )

    return U, all_converged, iters, runtime


def zbusjacobi(
    grid,
    S,
    U0=None,
    eps_s=1,
    max_iters=100,
    num_processes=1,
    sparse=False,
):
    starttime = time.perf_counter()

    grid_parameters = process_grid(grid, S)
    Zred = np.linalg.inv(grid_parameters["Yred"])
    if U0 is None:
        U0 = grid_parameters["u0"]
    slack_voltage = grid_parameters["main_slack_voltage"]
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    slack_index = grid_parameters["slack_index"]
    S = grid_parameters["S"]
    additional_slack_ids = grid_parameters["additional_slack_ids"]
    S[:, additional_slack_ids] = 0

    logger.debug(f"Setup Time: {time.perf_counter() - starttime:8.3f} s")

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
        U, iters = pf_cc.zbusjacobi_2slacks(
            Zred, S, U0, slack_voltage, i_slack, U_slack2, eps_s, eps_u_slack, max_iters
        )

    ## n Slacks:
    elif grid_parameters["numslacks"] > 2:
        m = len(grid_parameters["additional_slack_voltages"])
        U_additional_slacks = grid_parameters["additional_slack_voltages"]
        eps_u_slack = 0.001
        par_func = lambda S: pf_cc.zbusjacobi_mslacks(
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
        U, iters = _parallelize(par_func, S, num_processes)
    # 1 slack:
    else:
        if sparse:
            func = lambda S: powerflow_methods.zbusjacobi_sparse(
                Zred, S, slack_voltage, eps_s, max_iters
            )
        else:
            func = lambda S: pf_cc.zbusjacobi(
                Zred, S, U0, slack_voltage, eps_s, max_iters
            )
        U, iters = _parallelize(func, S, num_processes)

    # If slack nodes have been swapped to the top, swap them back so that the original order is restored:
    if grid_parameters["node_swaps"]:
        node_swaps = grid_parameters["node_swaps"].copy()
        while node_swaps:
            id1, id2 = node_swaps.pop()
            U[:, [id2, id1]] = U[:, [id1, id2]]

    all_converged = all(iters <= max_iters)

    runtime = time.perf_counter() - starttime

    _log_result(
        "Zbus Jacobi", runtime, np.mean(iters), np.min(np.abs(U)), all_converged
    )

    return U, all_converged, iters, runtime


def _bfs(grid, S, eps_s=1, max_iters=1000):
    starttime = time.perf_counter()
    grid_parameters = _parse_grid(grid)
    Zline = grid_parameters["Zline"]
    if Zline is None:
        name = "& Backward/Forward Sweep"
        print(f"{name:25s} & - & - \\\\")
        # print('Error: BFS is impossible because grid is not a feeder')
        return 0, 0, 0
    slack_voltage = grid_parameters["u0"][0]
    U, iters = pf_cc.bfs(Zline, S, slack_voltage, eps_s, max_iters)
    runtime = time.perf_counter() - starttime
    if verbose:
        _log_result(
            "Backward/Forward Sweep", runtime, np.mean(iters), np.min(np.abs(U))
        )
    return U, iters, runtime


def compare_methods(grid, S):
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Method")
    table.add_column("Iters", justify="right")
    table.add_column("Runtime", justify="right")

    pf_funcs = (
        ybusjacobi,
        ybusgaussseidel,
        ybusnewton,
        zbusjacobi,
    )

    set_loglevel("ERROR")

    with console.status("Computing power flow...") as status:
        for func in pf_funcs:
            U, all_converged, iters, runtime = func(grid, S)
            console.log(f"{func.__name__} done")
            table.add_row(func.__name__, str(np.mean(iters)), scale_time(runtime))

    console.print(table)
