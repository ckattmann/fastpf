import time
import numpy as np
import matplotlib.ticker

import powerflow
import powerflow.plotting as plt


def ybusjacobi(grid, S, eps_s=1.0, max_iters=10000):

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

    # Power Flow:
    for i in range(S.shape[0]):
        iters = 0
        s = S[i, :]

        while True:
            IR = Y @ U - np.conj(s / U)
            if np.max(np.abs(U * IR)) < eps_s or iters > max_iters:
                break
            U -= IR / diagY
            iters += 1

        u_all[i, :] = U
        iters_all[i] = iters

    # Output:
    runtime = time.time() - starttime
    iters = np.mean(iters_all)
    name = "YBus Jacobi"
    print(
        f"{name:20s}:  {runtime:.6f} s | {iters:6.0f} | {np.min(np.abs(u_all)):.3f} V"
    )

    return u_all, iters, runtime


def ybusjacobi_plot_each_iteration(grid, S, eps_s=1.0, max_iters=10000, plotting=True):

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

    # Init for Plotting:
    U_iters = [U.copy()]
    Smis_iters = [np.max(np.abs(S))]
    # Smis_max = 1000.0
    # Smis_min = 0.0

    # Power Flow:
    for i in range(S.shape[0]):
        iters = 0
        s = S[i, :]

        while True:
            IR = Y @ U - np.conj(s / U)
            if np.max(np.abs(U * IR)) < eps_s or iters > max_iters:
                break
            U -= 1.1 * (IR / diagY)
            iters += 1

            # Prepare Data for Plots:
            U_iters.append(U.copy())
            Smis = np.max(np.abs(U * IR))
            Smis_iters.append(Smis)
            if iters == 1:
                Smis_min = Smis - 1
                Smis_max = Smis + 1
            Smis_min = np.min([Smis_min, Smis])
            Smis_max = np.max([Smis_max, Smis])

            # Plotting:
            if plotting and len(U_iters) == 5:
                fig, axes = plt.subplots(len(U_iters), 1)
                plt.setsize(fig, 1)
                cmap = plt.get_cmap("BuGn")
                for ax in axes:
                    ax.ticklabel_format(useOffset=False)
                    ax.grid(True)
                    ax.set_ylim([397, 400])
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.FixedLocator([397, 398, 399, 400])
                    )
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.FixedFormatter(
                            ["397 V", "398 V", "399 V", "400 V"]
                        )
                    )
                    ax.yaxis.set_tick_params(labelsize=7)
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.FixedLocator([0, 1, 2, 3])
                    )
                    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
                    ax.xaxis.set_tick_params(length=0)
                for ax, U_i, Smis_i in zip(axes, U_iters, Smis_iters):
                    Smis_i_mapped = np.interp(Smis_i, [Smis_min, Smis_max], [1, 0.3])
                    ax.plot(
                        np.abs(U_i),
                        "-o",
                        color="green",
                        linewidth=2,
                        markersize=3,
                        zorder=10,
                        clip_on=False,
                    )
                    ax.plot(
                        [400, 399, 398, 397],
                        "--o",
                        color="lightgreen",
                        linewidth=1,
                        markersize=2,
                        zorder=5,
                        clip_on=False,
                    )
                # ax1.set_ylabel('Abs. Voltage / V')
                axes[-1].set_xlabel("Node no.")
                axes[-1].xaxis.set_major_locator(
                    matplotlib.ticker.FixedLocator([0, 1, 2, 3])
                )
                axes[-1].xaxis.set_major_formatter(
                    matplotlib.ticker.FixedFormatter([0, 1, 2, 3])
                )
                plt.gcf().text(
                    0.90,
                    0.9,
                    "0",
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                plt.gcf().text(
                    0.90,
                    0.725,
                    "1",
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                plt.gcf().text(
                    0.90,
                    0.55,
                    "2",
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                plt.gcf().text(
                    0.90,
                    0.375,
                    "3",
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                plt.gcf().text(
                    0.90,
                    0.2,
                    "4",
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                plt.gcf().text(
                    0.95,
                    0.5,
                    "$\leftarrow$ Iterations",
                    rotation=90,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                plt.tight_layout(0.3)
                plt.subplots_adjust(right=0.88)
                # plt.save(fig, 'ybusjacobi_iterations')
                plt.show()

        u_all[i, :] = U
        iters_all[i] = iters

    # Output:
    runtime = time.time() - starttime
    iters = np.mean(iters_all)
    name = "YBus Jacobi"
    print(
        f"{name:20s}:  {runtime:.6f} s | {iters:6.0f} | {np.min(np.abs(u_all)):.3f} V"
    )

    # # Plot the development of Smis:
    # if plotting:
    #     fig, ax2 = plt.subplots()
    #     plt.setsize(fig, 1)
    #     ax2.set_ylabel('Max. Abs. Power Residual / VA')
    #     ax2.set_xlabel('Iteration no.')
    #     ax2.plot(Smis_iters)
    #     ax2.grid(True)
    #     plt.tight_layout()
    #     plt.save(fig, 'ybusjacobi_iterations_Smis')
    #     plt.show()

    return u_all, iters, runtime


def ybusjacobi_plot_timings(grid, S, eps_s=1.0, max_iters=100000):

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
                IR = Y @ U - np.conj(s / U)
                if np.max(np.abs(U * IR)) < eps_s or iters > max_iters:
                    break
                U -= IR / diagY
                iters += 1

            u_all[i, :] = U
            iters_all[i] = iters

        solve_time = time.time() - starttime
        print(
            f"{n:2.0f} Nodes: {init_time * 1e6:5.0f} us + {int(solve_time / iters * 1e6):3.0f} us x {iters:5.0f} = {int(solve_time * 1e3):6.1f} ms"
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

U, iters, runtime = ybusjacobi_plot_each_iteration(grid, S, plotting=True)
# U, iters, runtime = ybusjacobi_plot_timings(grid,S)
