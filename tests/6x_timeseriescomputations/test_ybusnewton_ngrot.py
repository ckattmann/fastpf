import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def ybusnewton(Y, Yred, U, S, slack_index=0, eps_s=1, max_iters=100):

    numberofnodes = U.shape[0] - 1

    # This works, but looks stupid:
    non_slack_indices = np.zeros(numberofnodes, dtype=np.int32)
    counter = 0
    for i in range(S.shape[1]):
        if i != slack_index:
            non_slack_indices[counter] = i
            counter += 1

    # S = np.delete(S, deleted_indices, axis=1)
    # if slack_index == 0:
    #     non_slack_indices = np.arange(1, numberofnodes+1).astype(np.int64)
    # else:
    #     non_slack_indices = np.hstack((np.arange(0,slack_index).astype(np.int64), np.arange(slack_index+1, numberofnodes+1).astype(np.int64)))
    # print(non_slack_indices)

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

            diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            # dx = np.linalg.inv(-J) @ dPQ
            dx = np.linalg.solve(-J,dPQ)
            phi += dx[0:int(dx.shape[0]/2)]
            U_abs += (dx[int(dx.shape[0]/2):])

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all



if __name__ == '__main__':
    import powerflow
    import powerflow.data.mockgrids
    import matplotlib.pyplot as plt
    import time

    grid = powerflow.data.mockgrids.ngrot()
    # grid = powerflow.data.mockgrids.feeder(10, slack_index=5)
    # grid = powerflow.data.mockgrids.radial(100)

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    Y = grid_parameters['Y']
    Yred = grid_parameters['Yred']
    U0 = grid_parameters['u0']
    slack_index = grid_parameters['slack_index']
    deleted_indices = grid_parameters['deleted_node_ids']

    S = powerflow.data.mockloads.ngrot(grid, n=20)
    S = np.delete(S, deleted_indices, axis=1)

    # S = powerflow.data.mockloads.fixed(grid, load=1000)

    plt.plot(np.real(S.T))
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    starttime = time.time()
    U, iters = ybusnewton(Y, Yred, U0, S, slack_index=slack_index, eps_s=1, max_iters=100)
    print(time.time() - starttime)

    print('Iterations: ', iters)

    plt.plot(np.abs(U).T)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

