import time
import numpy as np
import powerflow
import powerflow.plotting as plt

def ybusgaussseidel(grid, S_all, acc_factor=1.0, eps_s=1, max_iters=20000, B=0.5):
    starttime = time.time()
    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    Y = grid_parameters['Y_ident']
    U = grid_parameters['u0']

    S_all = powerflow.calc_grid_parameters.integrate_slacks_for_Yident(S_all, grid)
    deleted_nodes = grid_parameters['node_ids_to_delete']
    S_all = np.delete(S_all, deleted_nodes, axis=1)

    u_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    diagY = np.diag(Y)
    for i in range(S_all.shape[0]):
        S = S_all[i,:]
        iters = 0
        S_mismatch_progression = []
        S_progression = []
        U_progression = []
        while True:
            S_mismatch = S - U * np.conj(Y @ U)
            S_mismatch_progression.append(S_mismatch.copy())

            S[1:] = S[1:] + B * S_mismatch[1:]

            S_progression.append(S.copy())
            for j in range(Y.shape[0]):
                U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(S[j] / U[j])) / diagY[j]
            U_progression.append(U.copy())

            # # Plot the progression of S and U:
            # fig, (ax1,ax2,ax3) = plt.subplots(3,1)
            # for S_mismatch_t in S_mismatch_progression:
            #     ax1.plot(S_mismatch_t[1:])
            # for S_t in S_progression:
            #     ax2.plot(S_t[1:])
            # for U_t in U_progression:
            #     ax3.plot(U_t[1:])
            # ax1.set_ylabel('Mismatch / VA')
            # ax1.grid(True)
            # ax2.set_ylabel('S + Boost / VA')
            # ax2.grid(True)
            # ax3.set_ylabel('Voltage / V')
            # ax3.grid(True)
            # plt.tight_layout()
            # plt.show()

            S_mismatch = U * np.conj(Y @ U) - S
            if np.max(np.abs(S_mismatch)) < eps_s or iters > max_iters:
                break
            iters += 1
        # Final iteration with true powers:
        S_mismatch = S - U * np.conj(Y @ U)
        S_mismatch_progression.append(S_mismatch.copy())
        S_progression.append(S.copy())
        for j in range(Y.shape[0]):
            U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(S[j] / U[j])) / diagY[j]
        U_progression.append(U.copy())

        # # Plot the progression of S and U:
        # fig, (ax1,ax2,ax3) = plt.subplots(3,1)
        # for S_mismatch_t in S_mismatch_progression:
        #     ax1.plot(np.abs(S_mismatch_t[1:]))
        # for S_t in S_progression:
        #     ax2.plot(np.abs(S_t[1:]))
        # for U_t in U_progression:
        #     ax3.plot(np.abs(U_t[1:]))
        # ax1.set_ylabel('Mismatch / VA')
        # ax1.grid(True)
        # ax2.set_ylabel('S + Boost / VA')
        # ax2.grid(True)
        # ax3.set_ylabel('Voltage / V')
        # ax3.grid(True)
        # plt.tight_layout()
        # plt.show()

        u_all[i,:] = U
        iters_all[i] = iters
    runtime = time.time() - starttime
    print(f'YBus Gauss-Seidel (B={B:.3f}):    {runtime:.6f} s | {np.mean(iters):6.0f} | {np.min(np.abs(U)):.3f} V')
    return U, iters, runtime

grid = powerflow.mockgrids.feeder(40)
S = powerflow.mockloads.fixed(grid, load=100, numberofloads=1)

# U, iters, runtime = ybusgaussseidel(grid, S, B=0)
for B in [-0.001, 0, 0.001, 0.005, 0.01, 0.02, 0.1]:
    U, iters, runtime = ybusgaussseidel(grid, S, B=B)
