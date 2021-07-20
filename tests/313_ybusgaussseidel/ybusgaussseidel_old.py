import time
import numba
import numpy as np

def ybusgaussseidel(Y, U, S_all, acc_factor=1.0, eps_s=1, max_iters=5000):
    ''' Standard YBus Gauss-Seidel Power Flow with optional acceleration factor
    '''
    u_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    diagY = np.diag(Y)
    for i in range(S_all.shape[0]):
        S = S_all[i,:]
        iters = 0
        while True:
            for j in range(Y.shape[0]):
                U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(S[j] / U[j])) / diagY[j]
            if np.max(np.abs(U * np.conj(Y @ U) - S)) < eps_s or iters > max_iters:
                break
            iters += 1
        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all


def pf_verbose(Y, U, S, acc_factor=1.0, i_eps=0.01, max_iters=4):

    np.set_printoptions(suppress=True)

    print(Y)
    diagY = np.diag(Y).copy()

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        s = S[i,:]
        iters = 0
        print()
        print('============== Starting YBusGaussSeidel ==============')
        print('Load: ',np.round(s,2))
        while True:
            print()
            print('===== Step ',iters,' =====')

            print('U : ', end='')
            print(np.round(U,2))

            print('Y@U : ', end='')
            print(np.round(Y@U,2))
            print('conj(s/U) : ', end='')
            print(np.conj(s/U))
            print('diag(Y)', end='')
            print(diagY)
            # print('R : ', end='')
            # print(R.T)

            old_U = U.copy()
            for j in range(Y.shape[0]):
                # print()
                # print('= Node '+str(j)+' =')
                # print('New Way:')
                # print('Y@U : ', np.round(Y[j,:]@U,2))
                # print('conj(s/U) : ', np.conj(s[j]/U[j]))
                # print('diag(Y) : ', diagY[j])
                # print('Ires : ', str((Y[j,:] @ U - np.conj(s[j] / U[j]))))
                # print('-> Step for node '+str(j)+': '+str(((Y[j,:] @ U) - np.conj(s[j] / U[j])) / diagY[j]))
                U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(s[j] / U[j])) / diagY[j]

            # Breaking Conditions

            # 2. Voltage Step
            print('max U change: ', np.max(np.abs(old_U-U)))
            if np.max(np.abs(old_U-U)) < 0.00001:
                break

            # 3. Power Residual
            print('Power Residual: ', np.round(s - U * np.conj(Y @ U),2))

            iters += 1
            if iters > max_iters:
                break

        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all


def pf_plotting(Y, U, S, acc_f0actor=1.0, u_eps=0.001, max_iters=2000):
    import matplotlib.pyplot as plt
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    diagY = np.diag(Y)
    for i in range(S.shape[0]):
        old_Us = U.copy()
        s = S[i,:]
        iters = 0
        while True:
            old_U = U.copy()
            for j in range(Y.shape[0]):
                U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(s[j] / U[j])) / diagY[j]
            if np.max(np.abs(old_U-U)) < u_eps or iters > max_iters:
                break
            iters += 1

            plt.title('Iteration '+str(iters))
            old_Us = np.vstack((old_Us, U))
            colors = plt.cm.Blues(np.linspace(0.1,1,old_Us.shape[0]))
            for u_index in range(old_Us.shape[0]):
                plt.plot(np.abs(old_Us[u_index,:]), color=colors[u_index])
            plt.grid(True)
            plt.show()
            U_prev = U.copy()

        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all


# Version with breaking condition R:
@numba.jit(nopython=True, cache=True)
def pf2(Y, U, S, acc_factor=1.0, i_eps=0.01, max_iters=5000):
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    diagY = np.diag(Y)
    for i in range(S.shape[0]):
        s = S[i,:]
        iters = 0
        while True:
            for j in range(Y.shape[0]):
                U[j] -= ((Y[j,:] @ U) - np.conj(s[j] / U[j])) / diagY[j]
            R = Y @ U - np.conj(s / U)
            if np.max(np.abs(R)) < i_eps or iters > max_iters:
                break
            iters += 1
        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all



@numba.jit(nopython=True, cache=True)
def ybusgaussseidel_cauchy(Y, U, S, acc_factor=1.0, u_eps=0.001, max_iters=5000):
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    diagY = np.diag(Y)
    for i in range(S.shape[0]):
        s = S[i,:]
        iters = 0
        while True:
            old_U = U.copy()
            for j in range(Y.shape[0]):
                U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(s[j] / U[j])) / diagY[j]
            if np.max(np.abs(old_U-U)) < u_eps or iters > max_iters:
                break
            iters += 1
        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all



def ybusgaussseidel_reversed(Y, U, S_all, acc_factor=1.0, eps_s=1, max_iters=5000):
    ''' Standard Gauss-Seidel Power Flow, reversed node order
        Especially for feeder grids, where the last node has 
        probably the highest voltage deviation
    ''' 
    u_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    diagY = np.diag(Y)
    for i in range(S_all.shape[0]):
        S = S_all[i,:]
        iters = 0
        while True:
            for j in reversed(range(Y.shape[0])):
                U[j] -= acc_factor * ((Y[j,:] @ U) - np.conj(S[j] / U[j])) / diagY[j]
            if np.max(np.abs(U * np.conj(Y @ U) - S)) < eps_s or iters > max_iters:
                break
            iters += 1
        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all

# @numba.jit(nopython=True, cache=True)
def ybusgaussseidel_autoacc(Y, U, S, acc_factor=1.0, eps_s=1, max_iters=100):
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    diagY = np.diag(Y)
    old_Imis = np.ones_like(U)
    Imis = np.zeros_like(U)
    for i in range(S.shape[0]):
        s = S[i,:]
        iters = 0
        while True:
            print()
            for j in range(Y.shape[0]):
                Imis[j] = (Y[j,:] @ U) - np.conj(s[j] / U[j])
                if iters > 0:
                    if Imis[j] < old_Imis[j]:
                        acc_factor = 1.0
                    else:
                        acc_factor = 1.4
                Ustep = Imis[j] / diagY[j]
                U[j] -= acc_factor * Ustep
                print(f'{j} : {acc_factor} {np.abs(Ustep):5.4f} -> {np.abs(U[j]):7.4f}, {np.abs(old_Imis[j]):4.3f} -> {np.abs(Imis[j]):4.3f}')
                old_Imis[j] = Imis[j]
            if np.max(np.abs(U * np.conj(Y @ U) - s)) < eps_s or iters > max_iters:
                break
            iters += 1
        u_all[i,:] = U
        iters_all[i] = iters
    return u_all, iters_all


if __name__ == '__main__':
    import powerflow

    grid = powerflow.mockgrids.feeder(40)
    S = powerflow.mockloads.beta(grid, maxload=2000)

    grid_parameters = powerflow.calc_grid_parameters.prepdata(grid)
    Y = grid_parameters['Y_ident']
    U0 = grid_parameters['u0']
    S = powerflow.calc_grid_parameters.integrate_slacks_for_Yident(S, grid)
    deleted_nodes = grid_parameters['node_ids_to_delete']
    S = np.delete(S, deleted_nodes, axis=1)


    # U_all, iters = ybusgaussseidel_autoacc(Y,U,S)

    U = U0.copy()
    U_all, iters_all = ybusgaussseidel(Y,U,S)
    print(f'YBus Gauss Seidel:  Umin={np.min(np.abs(U_all)):7.4f} V | Iters={np.mean(int(iters_all))}')

    U = U0.copy()
    U_all, iters_all = ybusgaussseidel_reversed(Y,U,S)
    print(f'YBus Gauss Seidel Reversed:  Umin={np.min(np.abs(U_all)):7.4f} V | Iters={np.mean(int(iters_all))}')
