import time
import numpy as np
import matplotlib.pyplot as plt
import numba

from powerflow.calc_grid_parameters import prepdata
import powerflow.data.mockgrids as mockgrids
import powerflow.data.mockloads as mockloads

import powerflow.powerflow_methods as pf_raw
import powerflow.powerflow as pf


VERBOSE = True
ACC_FACTOR = 1.0

@numba.jit()
def ybusnewton(Y, Yred, U, S, slack_index=0, eps_s=1, max_iters=100):

    numberofnodes = U.shape[0] - 1

    # This works, but looks stupid:
    non_slack_indices = np.zeros(numberofnodes, dtype=np.int32)
    counter = 0
    for i in range(S.shape[1]):
        if i != slack_index:
            non_slack_indices[counter] = i
            counter += 1

    S = S[:,non_slack_indices]

    U_abs = np.abs(U)[non_slack_indices]
    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dPQ = np.zeros(2*S.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes*2, numberofnodes*2), dtype=np.float64)

    Yredconj = np.conj(Yred)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j,:]

        U[non_slack_indices] = U_abs * np.exp(1j*phi)
        Scalc = U * (np.conj(Y @ U))
        Smismatch = Scalc[non_slack_indices] - s

        while iters <= max_iters:
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < eps_s:
                break
            iters += 1

            diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            dx = np.linalg.solve(-J,dPQ)

            phi +=  dx[0:int(dx.shape[0]/2)]
            U_abs += dx[int(dx.shape[0]/2):]
            U[non_slack_indices] = U_abs * np.exp(1j*phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


@numba.jit()
def ybusnewton_linesearch(Y, Yred, U, S, slack_index=0, eps_s=1, max_iters=100):

    numberofnodes = U.shape[0] - 1

    # This works, but looks stupid:
    non_slack_indices = np.zeros(numberofnodes, dtype=np.int32)
    counter = 0
    for i in range(S.shape[1]):
        if i != slack_index:
            non_slack_indices[counter] = i
            counter += 1

    S = S[:,non_slack_indices]

    U_abs = np.abs(U)[non_slack_indices]
    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dPQ = np.zeros(2*S.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes*2, numberofnodes*2), dtype=np.float64)

    Yredconj = np.conj(Yred)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j,:]

        U[non_slack_indices] = U_abs * np.exp(1j*phi)
        Scalc = U * (np.conj(Y @ U))
        Smismatch = Scalc[non_slack_indices] - s

        while iters <= max_iters:
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < eps_s:
                break
            iters += 1

            diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            dx = np.linalg.solve(-J,dPQ)

            # Line Search:
            phi_laststep = phi
            Uabs_laststep = U_abs
            alphas = np.arange(0.9,1.2,0.002)
            Smismatches = np.zeros(alphas.size)
            for i,alpha in enumerate(alphas):
                phi = phi_laststep + alpha * dx[0:int(dx.shape[0]/2)]
                U_abs = Uabs_laststep + alpha * (dx[int(dx.shape[0]/2):])
                U[non_slack_indices] = U_abs * np.exp(1j*phi)
                Scalc = U * (np.conj(Y @ U))
                Smismatch = Scalc[non_slack_indices] - s
                Smismatches[i] = np.mean(np.abs(Smismatch))
            best_alpha = alphas[np.argmin(Smismatches)]
            # print(best_alpha)

            # Apply the best alpha:
            phi = phi_laststep + best_alpha * dx[0:int(dx.shape[0]/2)]
            U_abs = Uabs_laststep + best_alpha * (dx[int(dx.shape[0]/2):])
            U[non_slack_indices] = U_abs * np.exp(1j*phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


if __name__ == "__main__":

    grid = mockgrids.meshed(200)
    S = mockloads.random(grid, maxload=1000, numberofloads=1)
    grid_parameters = prepdata(grid)

    Y = grid_parameters['Y']
    Yred = grid_parameters['Yred']
    U0 = grid_parameters['u0']
    slack_index = grid_parameters['slack_index']
    deleted_nodes = grid_parameters['deleted_node_ids']
    S = np.delete(S, deleted_nodes, axis=1)

    U1, iters = ybusnewton(Y, Yred, U0, S, slack_index, eps_s=1, max_iters=100)
    U2, iters = ybusnewton_linesearch(Y, Yred, U0, S, slack_index, eps_s=1, max_iters=100)
    U, iters, runtime = pf.ybusnewton(grid_parameters,S)
    print('-----------')

    starttime = time.time()
    U, iters = ybusnewton_linesearch(Y, Yred, U0, S, slack_index, eps_s=1, max_iters=100)
    runtime = time.time() - starttime
    if VERBOSE:
        print('\ybus Newton Line Search',runtime, np.mean(iters), np.min(np.abs(U)))

    starttime = time.time()
    U, iters = ybusnewton(Y, Yred, U0, S, slack_index, 1, 100)
    runtime = time.time() - starttime
    if VERBOSE:
        print('\ybus Newton Local',runtime, np.mean(iters), np.min(np.abs(U)))

    starttime = time.time()
    U, iters = pf_raw.ybusnewton(Y, Yred, U0, S, slack_index, 1, 100)
    runtime = time.time() - starttime
    if VERBOSE:
        print('\ybus Newton Raw',runtime, np.mean(iters), np.min(np.abs(U)))

    starttime = time.time()
    U, iters, runtime2 = pf.ybusnewton(grid_parameters,S)
    runtime = time.time() - starttime
    if VERBOSE:
        print('\ybus Newton',runtime, np.mean(iters), np.min(np.abs(U)))


