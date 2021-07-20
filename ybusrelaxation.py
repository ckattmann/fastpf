import numpy as np
import time
import numba


def prepdata(grid, linesToRemove=[], baseVoltage=None):

    nodes = grid['nodes']
    lines = grid['edges']
    numberofnodes = len(nodes)
    numberoflines = len(lines)

    if not baseVoltage:
        slack_voltages = []
        for n in nodes:
            if 'slack_voltage' in n:
                slack_voltages.append(n['slack_voltage'])
        if slack_voltages:
            baseVoltage = np.mean(slack_voltages)
        else:
            print('No Voltages for slack found')
            return
    u0 = np.ones(numberofnodes, dtype=np.complex128) * baseVoltage

    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)

    # Line Impedances Matrix: Used later for current calculation
    line_impedances = np.zeros((numberofnodes, numberoflines), dtype=np.complex128)

    # Add the connections in 'lines' to the admittancy matrix Y
    for i, line in enumerate(lines):
        if line['id'] not in linesToRemove:

            # Make sure connection goes from lower node number to higher node number,
            # for easy symmetrization with np.transpose later
            if line['source'] > line['target']:
                from_node, to_node = line['target'], line['source']
            else:
                from_node, to_node = line['source'], line['target']

            # Make the connection in the admittancy matrix
            Y[from_node, to_node] += 1 / (line['R'] + 1j * line['X'])

            # Make the connection in the line_impedances matrix
            line_impedances[from_node, i] += 1 / (line['R'] + 1j * line['X'])
            line_impedances[to_node, i] += -1 / (line['R'] + 1j * line['X'])
        else:
            pass
            # print('Line '+line['name']+', number '+str(line['line_number'])+' removed')

    # Symmetrize
    Y = Y + np.transpose(Y)

    # Assign diagonal elements
    for i in range(0,len(Y)):
        Y[i,i] = -np.sum(Y[i,:])

    # Change lines of slacks to idents
    for i, node in enumerate(nodes):
        if node['is_slack']:
            Y[i,:] = np.zeros_like(Y[i,:])
            Y[i,i] = 1
            u0[i] = node['slack_voltage'] * np.exp(1j*node['slack_angle']/360*2*np.pi)

    pf_parameters = (Y, u0)
    extra_parameters = (line_impedances)

    return pf_parameters, extra_parameters


# @numba.jit(nopython=True, cache=True)
# @numba.jit(cache=True)
def pf_verbose(Y, U, S):
    print(Y)
    i = 0
    diagY = np.diag(Y).copy()

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        s = S[i,:]
        iters = 0
        print('Starting YBusRelaxation')
        print('=======================')
        print('Load: ',np.round(s,2))
        print()
        while True:
            print()
            print('================ Step ',iters,' ================')
            R = Y @ U - np.conj(s / U)
            # print('U : ', end='')
            # print(np.round(U,2))
            # print('Y@U : ', end='')
            # print(np.round(Y@U,2))
            # print('s/U : ', end='')
            # print(s/U)
            # print('R : ', end='')
            # print(R.T)
            max_R_index = np.argmax(np.abs(R))
            print('maxR : ',np.round(R[max_R_index], 2))
            max_R = R[max_R_index]

            if np.abs(max_R) < 0.01:
                break
            iters += 1

            print('R : ',np.round(R.T,2))

            dU = R[max_R_index] / diagY[max_R_index]
            U[max_R_index] -= dU

            Ralt = R.copy()
            ue = np.zeros_like(U)
            ue[max_R_index] = dU
            Ralt = R + Y @ ue
            print('Ralt : ')
            print(np.round(Ralt.T,2))

            if iters > 20:
                break

        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all


def pf_plot(Y, U, S):
    import matplotlib.pyplot as plt

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        u_plot = np.zeros((1000, U.shape[0]), dtype=np.complex128)  # 1000 is guessed as maximum number of iterations
        s = S[i,:]
        iters = 0
        maxR_progress = []
        # Plot U0
        fig, ax = plt.subplots()
        ax.set_title('Step '+str(iters)+', U0')
        ax.plot(U)
        plt.show()
        while True:
            R = Y @ U - np.conj(s / U)
            max_R_index = np.argmax(np.abs(R))
            maxR_progress.append(R[max_R_index])
            if np.abs(R[max_R_index]) < 0.01:
                break
            U[max_R_index] -= R[max_R_index] / Y[max_R_index, max_R_index]
            u_plot[iters,:] = U
            iters += 1
            print('U = ',np.round(u_plot[iters-1,:], 2))
            # Plotting:
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.set_title('Step '+str(iters)+', R='+str(np.max(np.abs(R))))
            ax1.plot(u_plot[:iters,:].T)
            ax1.grid(True)
            ax2.semilogy(np.abs(maxR_progress), 'o-')
            ax2.grid(True)
            plt.show()

        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all




@numba.jit(nopython=True, cache=True)
def pf(Y, U, S, acc_factor=1.0, i_eps=0.01, max_iters=20000):
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        s = S[i,:]
        iters = 0
        while True:
            R = Y @ U - np.conj(s / U)
            max_R_index = np.argmax(np.abs(R))
            if np.abs(R[max_R_index]) < i_eps or iters > max_iters:
                break
            iters += 1
            U[max_R_index] -= acc_factor * R[max_R_index] / Y[max_R_index, max_R_index]
        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all


if __name__ == '__main__':
    import data.mockgrids
    import data.mockloads
    import calc_grid_parameters
    import matplotlib.pyplot as plt

    grid = data.mockgrids.feeder(5)
    S = data.mockloads.fixed(grid, load = 5000)

    grid_parameters = calc_grid_parameters.prepdata(grid)
    Y = grid_parameters['Y']
    u0 = grid_parameters['u0']
    # U, iters = pf(Y, u0, S, 1.0, 0.00001, 10000)
    # print(iters)
    U, iters = pf_plot_through(Y, u0, S)
    print(iters)

    plt.plot(U.T)
    plt.grid()
    plt.show()
