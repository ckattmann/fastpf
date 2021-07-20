import time
import numpy as np
import powerflow
import powerflow.plotting as plt

def ybusnewton(grid, S, eps_s=1, max_iters=10, num_processes=1, sparse=False, verbose=True):
    starttime = time.time()
    grid_parameters = powerflow.calc_grid_parameters.calc_grid_parameters(grid, S, reorder_slack_nodes=True)
    Y = grid_parameters['Y']
    U0 = grid_parameters['u0']
    num_slacks = grid_parameters['numslacks']
    additional_slack_voltages = grid_parameters['additional_slack_voltages']
    # additional_slack_ids = grid_parameters['additional_slack_ids']
    for i,u_slack in enumerate(additional_slack_voltages,1):
        S[:,i] = u_slack
    for i,u_slack in enumerate(additional_slack_voltages,1):
        U0[i] = u_slack
    U, iters = ybusnewton_slack0(Y, S, U0, num_slacks, eps_s, max_iters)

    if grid_parameters['node_swaps']:
        node_swaps = grid_parameters['node_swaps'].copy()
        while node_swaps:
            id1, id2 = node_swaps.pop()
            U[:,[id2, id1]] = U[:,[id1, id2]]

    runtime = time.time() - starttime
    if verbose:
        print_result('Ybus Newton',runtime, np.mean(iters), np.min(np.abs(U)))
    return U, iters, runtime



# @cc.export('ybusnewton','(complex128[:,:], complex128[:,:], complex128[:], int64, float64, int64)')
def ybusnewton_mslacks(Y, S_all, U, m, eps_s=1, max_iters=100):
    ''' YBus-Newton Power Flow for multiple slack nodes
        Y       [n x n]     complex128: Admittance matrix, main_slack in first line, other slacks in subsequent lines
        S_all   [l x n]     complex128: Loads in Watt
        U       [n]         complex128: Starting voltages, slack voltages need to be in place
        m                   int64:      Number of additional slacks
        eps_s               float64:    power residual
        max_iters           int64:      number of iterations before break
    '''

    numberofnodes = U.shape[0] - 1

    S_all = S_all[:,1:]

    U_abs = np.abs(U)[1:]
    phi = np.zeros(U_abs.size, dtype=np.float64)

    U_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    additional_slack_voltages = U[1:m]
    dPQ = np.zeros(2*S_all.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes*2, numberofnodes*2), dtype=np.float64)

    Yredconj = np.conj(Y[1:,1:])

    iters = 0

    for j in range(S_all.shape[0]):
        iters = 0
        s = S_all[j,:]

        while iters <= max_iters:
            U[1:] = U_abs * np.exp(1j*phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[1:] - s
            dPQ = np.hstack((Smismatch.real, Smismatch.imag))
            for i in range(len(additional_slack_voltages)):
                dPQ[i] = 0
                dPQ[i+numberofnodes] = 0
            if np.max(np.abs(dPQ)) < eps_s:
                break

            diagIconj = np.diag(np.conj((Y @ U)[1:]))
            diagUred = np.diag(U[1:])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            # Handle multiple slacks:
            for i in range(len(additional_slack_voltages)):
                J[i,:] = np.zeros(numberofnodes*2)
                J[i,i] = 1
                i_phase = i + numberofnodes
                J[i_phase,:] = np.zeros(numberofnodes*2)
                J[i_phase,i_phase] = 1

            dx = np.linalg.solve(-J,dPQ)
            phi += dx[0:int(dx.shape[0]/2)]
            U_abs += (dx[int(dx.shape[0]/2):])

            iters += 1

        U_all[j] = U
        iters_all[j] = iters

    return U_all, iters_all


def print_result(name, runtime, mean_iters, min_U):
    runtime_ms = runtime * 1000
    print(f'{name:20s}:  {runtime:.6f} s | {mean_iters:6.0f} | {min_U:.3f} V')


if __name__ == '__main__':
    grid = powerflow.grids.radial(6, num_slacks=3, slack_voltages=[400,390,380], R=1)
    S = powerflow.loads.random(grid, maxload=1000, numberofloads=1)

    U, iters, runtime = ybusnewton(grid, S)

    plt.plot(U.T, 'o-')
    plt.grid(True)
    plt.show()

