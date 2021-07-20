import time
import pickle
import numpy as np
import scipy.sparse
import numba
import matplotlib.pyplot as plt

import powerflow
import powerflow.plotting as plt

# @numba.jit(nopython=True, cache=True)
def ybusnewton_csc(Y, Yred, U, S, slack_index=0, eps_s=1, max_iters=100):

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

    # Convert to sparse:
    Y = scipy.sparse.csc_matrix(Y)

    for j in range(S.shape[0]):
        iters = 0
        s = S[j,:]

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j*phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < eps_s:
                break
            iters += 1

            diagIconj = np.diag(np.conj((Y.dot(U))[non_slack_indices]))
            # diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            # J = np.zeros((numberofnodes*2, numberofnodes*2), dtype=np.float64)
            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            J = scipy.sparse.csc_matrix(J)
            # dx = np.linalg.solve(-J,dPQ)
            dx = scipy.sparse.linalg.spsolve(-J,dPQ)
            phi += dx[0:int(dx.shape[0]/2)]
            U_abs += (dx[int(dx.shape[0]/2):])

            J = J.todense()

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


def ybusnewton_csr(Y, Yred, U, S, slack_index=0, eps_s=1, max_iters=100):

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

    # Convert to sparse:
    Y = scipy.sparse.csr_matrix(Y)

    for j in range(S.shape[0]):
        iters = 0
        s = S[j,:]

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j*phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < eps_s:
                break
            iters += 1

            diagIconj = np.diag(np.conj((Y.dot(U))[non_slack_indices]))
            # diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            # J = np.zeros((numberofnodes*2, numberofnodes*2), dtype=np.float64)
            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            J = scipy.sparse.csr_matrix(J)
            # dx = np.linalg.solve(-J,dPQ)
            dx = scipy.sparse.linalg.spsolve(-J,dPQ)
            phi += dx[0:int(dx.shape[0]/2)]
            U_abs += (dx[int(dx.shape[0]/2):])

            J = J.todense()

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all



def sim():
    runtimes_nonsparse = []
    runtimes_sparse_csr = []
    runtimes_sparse_csc = []
    nodes = np.arange(10,101,10)
    # nodes = [10,20,30,40,50,60,70,80,90,100,150,200]

    for n in nodes:
        print()
        print(f'{n} nodes:')
        grid = powerflow.mockgrids.meshed(n)
        S = powerflow.mockloads.beta(grid, numberofloads=350)

        grid_parameters = powerflow.prepdata(grid)
        Y = grid_parameters['Y']
        Yred = grid_parameters['Yred']
        u0 = grid_parameters['u0']
        Zred = np.linalg.inv(grid_parameters['Yred'])
        deleted_nodes = grid_parameters['deleted_node_ids']
        slack_index = grid_parameters['slack_index']

        U, iters, runtime = powerflow.ybusnewton(grid_parameters, S, verbose=False)
        runtimes_nonsparse.append(runtime)
        print(f'Runtime non-sparse: {runtime*1000:.3f} ms')

        starttime = time.time()
        U, iters = ybusnewton_csr(Y, Yred, u0, S, slack_index=0)
        runtimes_sparse_csr.append(time.time() - starttime)
        print(f'Runtime sparse CSR: {(time.time() - starttime) * 1000:.3f} ms')

        starttime = time.time()
        U, iters = ybusnewton_csc(Y, Yred, u0, S, slack_index=0)
        runtimes_sparse_csc.append(time.time() - starttime)
        print(f'Runtime sparse CSC: {(time.time() - starttime) * 1000:.3f} ms')

    plotdata = {'nodes': nodes, 
            'runtimes_nonsparse': runtimes_nonsparse, 
            'runtimes_sparse_csc': runtimes_sparse_csc, 
            'runtimes_sparse_csr': runtimes_sparse_csr}

    with open('sparse_ybus_newton.simdata','wb') as f:
        pickle.dump(plotdata,f)

    return nodes, runtimes_nonsparse, runtimes_sparse_csc, runtimes_sparse_csr


if __name__ == '__main__':

    resim = False

    if resim:
        nodes, runtimes_nonsparse, runtimes_sparse_csc, runtimes_sparse_csr = sim()
    else:
        with open('sparse_ybus_newton.simdata','rb') as f:
            nodes, runtimes_nonsparse, runtimes_sparse_csc, runtimes_sparse_csr = pickle.load(f)

    fig, ax = plt.subplots()
    fig.set_size_inches((4.33,2.5))

    ax.plot(nodes,runtimes_nonsparse, '.-', markersize=6, label='non-sparse')
    ax.plot(nodes,runtimes_sparse_csr, '.-', markersize=6, label='CSR')
    ax.plot(nodes,runtimes_sparse_csc, '.-', markersize=6, label='CSC')
    ax.set_xlabel('Number of nodes in grid / -')
    ax.set_ylabel('Runtime / s')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.save(fig,'sparseYbusNewton')
    plt.show()

