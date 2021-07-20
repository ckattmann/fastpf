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


def ybusnewton_oneiteration(Y, s, U):

    numberofnodes = len(U) - 1
    U_abs = np.abs(U)[1:]
    phi = np.zeros(U_abs.size, dtype=np.float64)
    J = np.zeros((numberofnodes * 2, numberofnodes * 2), dtype=np.float64)
    Yredconj = np.conj(Y[1:, 1:])

    s = s[1:]
    eps_s = 1

    U[1:] = U_abs * np.exp(1j * phi)
    Scalc = U * (np.conj(Y @ U))
    Smismatch = Scalc[1:] - s
    Pcalc = Smismatch.real
    Qcalc = Smismatch.imag
    dPQ = np.hstack((Pcalc, Qcalc))
    if np.max(np.abs(dPQ)) < eps_s:
        print("Convergence")
        # break

    diagIconj = np.diag(np.conj((Y @ U)[1:]))
    diagUred = np.diag(U[1:])
    Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
    JU = diagIconj + diagUred @ Yredconj

    J[:numberofnodes, :numberofnodes] = np.real(Jphi)
    J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
    J[:numberofnodes, numberofnodes:] = np.real(JU)
    J[numberofnodes:, numberofnodes:] = np.imag(JU)

    dx = np.linalg.solve(-J, dPQ)
    phi += dx[0 : int(dx.shape[0] / 2)]
    U_abs += dx[int(dx.shape[0] / 2) :]
    U[1:] = U_abs * np.exp(1j * phi)
    return U


def zbusjacobi_oneiteration(Zred, s, U):
    Ibus = np.conj(s / U)
    U = Zred @ Ibus + slack_voltage
    return U


def plot_each_iteration(
    grid,
    S,
    iter_func,
    maxiters,
    plot_filename="",
    eps_s=1.0,
    max_iters=10000,
    acc_factor=1.0,
    plotting=True,
):

    # Preparation & Intialization:
    starttime = time.time()
    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    Y = grid_parameters["Y_ident"]
    Yred = grid_parameters["Yred"]
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

    # Power Flow:
    for i in range(S.shape[0]):
        s = S[i, :]

        iters = 0
        for k in range(maxiters):
            # U = ybusnewton_oneiteration(Y,s,U)
            U = ybusrelaxation_oneiteration(Y, s, U)
            # U = iter_func(Y,s,U)
            # if np.max(np.abs(U * np.conj(Y @ U) - s)) < eps_s or iters > max_iters:
            #     break
            iters += 1

            # Prepare Data for Plots:
            U_iters.append(U.copy())
            Smis = np.max(np.abs(U * np.conj(Y @ U) - s))
            Smis_iters.append(Smis)
            if iters == 1:
                Smis_min = Smis - 1
                Smis_max = Smis + 1
            Smis_min = np.min([Smis_min, Smis])
            Smis_max = np.max([Smis_max, Smis])

            # Plotting:
            if plotting and iters == maxiters:
                fig, axes = plt.subplots(len(U_iters), 1)
                if maxiters == 1:
                    plt.setsize(fig, 0.5)
                elif maxiters == 4:
                    plt.setsize(fig, 1)
                cmap = plt.get_cmap("BuGn")
                for ax in axes:
                    ax.ticklabel_format(useOffset=False)
                    ax.grid(True, color="lightgray")
                    ax.set_ylim([397, 400])
                    ax.set_xlim([0, 3])
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.FixedLocator([397, 398, 399, 400])
                    )
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.FixedFormatter(
                            ["397 V", "398 V", "399 V", "400 V"]
                        )
                    )
                    ax.yaxis.set_tick_params(labelsize=7)
                    ax.tick_params(axis="y", colors="lightgray")
                    plt.setp(ax.get_yticklabels(), color="black")
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.FixedLocator([0, 1, 2, 3])
                    )
                    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
                    ax.xaxis.set_tick_params(length=0)
                    ax.spines["bottom"].set_color("lightgray")
                    ax.spines["top"].set_color("lightgray")
                    ax.spines["right"].set_color("lightgray")
                    ax.spines["left"].set_color("lightgray")
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
                axes[-1].set_xlabel("Node number")
                axes[-1].xaxis.set_major_locator(
                    matplotlib.ticker.FixedLocator([0, 1, 2, 3])
                )
                axes[-1].xaxis.set_major_formatter(
                    matplotlib.ticker.FixedFormatter([0, 1, 2, 3])
                )
                if maxiters == 4:
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
                elif maxiters == 1:
                    plt.gcf().text(
                        0.90,
                        0.79,
                        "0",
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                    plt.gcf().text(
                        0.90,
                        0.41,
                        "1",
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                    plt.gcf().text(
                        0.95,
                        0.6,
                        "$\leftarrow$ Iterations",
                        rotation=90,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                plt.tight_layout(0.3)
                plt.subplots_adjust(right=0.88)
                if plot_filename:
                    plt.save(fig, plot_filename)
                plt.show()
                break

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


grid = powerflow.mockgrids.feeder(4, R=1)
S = powerflow.mockloads.fixed(grid, 0, 1)
S[:, -1] = 400

U, iters, runtime = plot_each_iteration(
    grid,
    S,
    iter_func=ybusjacobi_oneiteration,
    maxiters=4,
    plot_filename="ybusjacobi_iterations",
    plotting=True,
)
U, iters, runtime = plot_each_iteration(
    grid,
    S,
    iter_func=ybusgaussseidel_oneiteration,
    maxiters=4,
    plot_filename="ybusgaussseidel_iterations",
    plotting=True,
)
# U, iters, runtime = plot_each_iteration(grid,S, iter_func=ybusrelaxation_oneiteration, maxiters=4, plot_filename='ybusrelaxation_iterations', plotting=True)
# U, iters, runtime = plot_each_iteration(grid,S, iter_func=ybusnewton_oneiteration, maxiters=1, plot_filename='ybusnewton_iterations', plotting=True)
