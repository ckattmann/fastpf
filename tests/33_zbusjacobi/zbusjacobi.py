import time
import numpy as np

import powerflow
import powerflow.plotting as plt


def zbusjacobi(grid, S, sparse=False):
    starttime = time.time()
    eps_s = 1
    max_iters = 10
    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    Y = grid_parameters["Y"]
    Y[0, 0] -= 10000
    # Y[-1,-1] -= 10000
    Z = np.linalg.inv(Y)
    Z[0, :] = np.zeros(Z.shape[0])
    Z[0, 0] = 1

    # Z[-1,:] = np.zeros(Z.shape[0])
    # Z[-1,-1] = 1

    U0 = grid_parameters["u0"]
    U0 = grid_parameters["u0"]
    slack_voltage = U0[0]
    # deleted_nodes = grid_parameters['node_ids_to_delete']
    # slack_index = grid_parameters['slack_index']
    # S = np.delete(S, deleted_nodes, axis=1)
    # S = powerflow.calc_grid_parameters.integrate_slacks_for_Yident(S, grid)
    S[:, 0] = np.zeros(S.shape[0])
    S[:, -1] = np.zeros(S.shape[0])
    # print(S)
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

    def plot(x, c="b"):
        plt.plot(np.sign(x) * np.abs(x), c + "o-")
        plt.grid()
        plt.show()

    for i in range(numberofloads):
        s = S[i, :]
        iters = 0
        while True:
            # print('U',U)
            Ibus = np.conj(s / U)

            Us2 = 10 + 10j
            Us2 -= np.sum(Z[-1, :-1] @ Ibus[:-1])
            Is2 = Us2 / Z[-1, -1]
            Ibus[-1] = Is2
            s[-1] = np.conj(Ibus[-1]) * U[-1]
            # plot(s,'k')

            U = Z @ Ibus + slack_voltage
            iters += 1
            # if np.max(np.abs(s - U * np.conj(Ibus))) < eps_s or iters > max_iters:
            #     break
            # print('epsu', np.abs(U[-1] - (slack_voltage + 10+10j)))
            # print('eps_s', np.max(np.abs(s - U * np.conj(Ibus))))
            if (
                np.max(np.abs(s - U * np.conj(Ibus))) < eps_s
                and np.abs(U[-1] - (slack_voltage + 10 + 10j)) < 0.01
            ):
                break
            if iters > max_iters:
                break
            # plot(U, 'g')

            # print(np.abs(U))

        U_all[i, :] = U
        iters_all[i] = iters

    runtime = time.time() - starttime

    name = "ZBus Jacobi"
    print(
        f"{name:20s}:  {runtime:.6f} s | {np.mean(iters_all):6.0f} | {np.min(U):.3f} V"
    )

    return U_all, iters_all


def zbusjacobi_reduced(grid, S, sparse=False):
    """Uses the reduced version of the Y with deleted slack"""
    starttime = time.time()
    eps_s = 1
    max_iters = 10
    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    Zred = np.linalg.inv(grid_parameters["Yred"])
    U0 = grid_parameters["u0"]
    slack_voltage = U0[0]
    deleted_nodes = grid_parameters["node_ids_to_delete"]
    slack_index = grid_parameters["slack_index"]
    S = np.delete(S, deleted_nodes + [slack_index], axis=1)
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    for i in range(numberofloads):
        s = S[i, :]
        iters = 0
        while True:
            Ibus = np.conj(s / U)
            U = Zred @ Ibus + slack_voltage
            iters += 1
            if np.max(np.abs(s - U * np.conj(Ibus))) < eps_s or iters > max_iters:
                break
        U_all[i, :] = U
        iters_all[i] = iters
    runtime = time.time() - starttime
    name = "ZBus Jacobi Reduced"
    print(
        f"{name:20s}:  {runtime:.6f} s | {np.mean(iters_all):6.0f} | {np.min(U):.3f} V"
    )
    return U_all, iters_all


grid = powerflow.mockgrids.feeder(4, R=1, X=1)
S = powerflow.mockloads.fixed(grid, 0, 1)
S[:, -2] = 400

U, iters = zbusjacobi(grid, S.copy())
U, iters = zbusjacobi_reduced(grid, S.copy())
