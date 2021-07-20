import os
import time
import numba
from numba.pycc import CC
import numpy as np

import matplotlib.pyplot as plt

# os.environ['NUMBA_DEBUG_CACHE'] = '1'

cc = CC('ybusjacobi_cc')

def pf_verbose(Y, U, S, acc_factor=1.0, i_eps=0.001, max_iters=4):
    print('\n------> YBus Jacobi -------------')
    print('Y:',Y)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    for i in range(S.shape[0]):
        iters = 0
        s = S[i,:]
        while True:
            print('U : ', U)
            R = Y @ U - np.conj(s / U)
            print('====== Iteration '+str(iters)+' =====')
            print('S : ', s)
            print('Y@U : ', Y@U)
            print('S/U : ', S/U)
            print('R : ',R)

            if np.max(np.abs(R)) < i_eps or iters > max_iters:
                break

            dU = R / np.diag(Y)
            U -= acc_factor * dU
            iters += 1

        u_all[i,:] = U
        iters_all[i] = iters

    return u_all, iters


def pf_plotting(Y, U, S, acc_factor=1.0, i_eps=0.001, max_iters=10000):
    diagY = np.diag(Y)
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        iters = 0
        s = S[i,:]
        U_all = []
        while True:
            R = Y @ U - np.conj(s / U)
            if np.max(np.abs(R)) < i_eps or iters > max_iters:
                break
            iters += 1
            U -= acc_factor * R / diagY

            U_all.append(U.copy())
            colors = plt.cm.Blues(np.linspace(0.1,1,len(U_all)))
            for u_index,u in enumerate(U_all):
                plt.plot(u, color=colors[u_index])
            plt.grid(True)
            plt.show()

        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all


def ybusjacobi_eps_i(Y, U, S, acc_factor=1.0, eps_i=0.01, max_iters=10000):
    Y = Y.copy()
    S = S.copy()
    diagY = np.diag(Y)
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        iters = 0
        s = S[i,:]
        while True:
            R = Y @ U - np.conj(s / U)
            if np.max(np.abs(R)) < eps_i or iters > max_iters:
                break
            iters += 1
            U -= acc_factor * R / diagY
        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all


def ybusjacobi_eps_u(Y, U, S, acc_factor=1.0, eps_u=0.001, max_iters=10000):
    Y = Y.copy()
    S = S.copy()
    diagY = np.diag(Y).copy()
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if i != j:
                Y[i,j] /= Y[i,i]
        Y[i,i] = 0
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)
    U_old = U.copy()
    for i in range(S.shape[0]):
        iters = 0
        s = np.conj(S[i,:]) / diagY
        while True:
            U = s / np.conj(U) - Y@U
            if iters > 10:
                if np.max(np.abs(U - U_old)) < eps_u or iters > max_iters:
                    break
                U_old = U.copy()
            iters += 1
        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all


def ybusjacobi_eps_s(Y, U, S, acc_factor=1.0, eps_s=0.01, max_iters=10000):
    Y = Y.copy()
    S = S.copy()
    diagY = np.diag(Y)
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        iters = 0
        s = S[i,:]
        while True:
            YU = Y @ U
            dS = U * np.conj(YU) - s
            if np.max(np.abs(dS)) < eps_s or iters > max_iters:
                break
            iters += 1
            U -= (YU - np.conj(s / U)) / diagY
        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all

def ybusjacobi_eps_s2(Y, U, S, acc_factor=1.0, eps_s=0.01, max_iters=10):
    Y = Y.copy()
    S = S.copy()
    diagY = np.diag(Y)
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        iters = 0
        s = S[i,:]
        while True:
            IR = Y @ U - np.conj(s / U)
            dS = U * IR
            if np.max(np.abs(dS)) < eps_s or iters > max_iters:
                break
            iters += 1
            U -= IR / diagY
        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all



def ybusjacobi_eps_u_2(Y, U, S, acc_factor=1.0, eps_u=0.001, max_iters=10000):
    Y = Y.copy()
    S = S.copy()
    diagY = np.diag(Y).copy()
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if i != j:
                Y[i,j] /= Y[i,i]
        Y[i,i] = 0
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)
    U_old = U.copy()
    S = np.conj(S) / diagY
    for i in range(S.shape[0]):
        iters = 0
        # s = np.conj(S[i,:]) / diagY
        s = S[i,:]
        while True:
            U = s / np.conj(U) - Y@U
            if iters > 10:
                if np.max(np.abs(U - U_old)) < eps_u or iters > max_iters:
                    break
                U_old = U.copy()
            iters += 1
        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all


# @numba.jit(nopython=True, cache=True)
def pf2_verbose(Y, U, S, acc_factor=1.0, u_eps=0.1, max_iters=10000):
    max_iters = 10
    diagY = np.diag(Y).copy()
    print('Ydiag:',diagY)
    print('Y:\n ',Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            print(i,j,Y[i,j])
            if i != j:
                Y[i,j] /= Y[i,i]
            print(i,j,Y[i,j])
        Y[i,i] = 0
    print('Y after transformation:\n',Y)
    u_result = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)
    U_old = U.copy()
    for i in range(S.shape[0]):
        iters = 0
        print('Ydiag: ',diagY)
        s = np.conj(S[i,:]) / diagY
        print('s:',s)
        print('U:',U)
        while True:
            U = s / np.conj(U) - Y@U
            print('U:',U)
            print('Udiff: ',np.max(np.abs(U - U_old)))
            if np.max(np.abs(U - U_old)) < u_eps or iters > max_iters:
                break
            U_old = U.copy()
            iters += 1
        u_result[i,:] = U
        iters_all[i] = iters
    return u_result, iters_all


if __name__ == '__main__':

    import data.mockgrids
    import data.mockloads
    import calc_grid_parameters

    grid = data.mockgrids.feeder(10)
    S = data.mockloads.random(grid, maxload=5000, numberofloads=10000)

    grid_parameters = calc_grid_parameters.prepdata(grid)

    Y = grid_parameters['Y']
    u0 = grid_parameters['u0']

    # starttime = time.time()
    # U, iters = pf_verbose(*pf_parameters,S)
    # print(np.min(np.abs(U)), np.max(np.abs(U)), np.mean(iters), time.time() - starttime)

    # starttime = time.time()
    # U, iters = ybusjacobi_eps_u(Y,u0,S, eps_u=0.00001)
    # # print(U)
    # print(np.min(np.abs(U)), np.max(np.abs(U)), np.mean(iters), time.time() - starttime)

    # starttime = time.time()
    # U, iters = ybusjacobi_eps_u_2(Y,u0,S,eps_u=0.00001)
    # # print(U)
    # print(np.min(np.abs(U)), np.max(np.abs(U)), np.mean(iters), time.time() - starttime)

    # starttime = time.time()
    # U, iters = ybusjacobi_eps_i(Y,u0,S,eps_i=0.01)
    # print(np.min(np.abs(U)), np.max(np.abs(U)), np.mean(iters), time.time() - starttime)

    starttime = time.time()
    U, iters = ybusjacobi_eps_s(Y,u0,S,eps_s=0.01)
    print(np.min(np.abs(U)), np.max(np.abs(U)), np.mean(iters), time.time() - starttime)

    # starttime = time.time()
    # U, iters = ybusjacobi_eps_s2(Y,u0,S,eps_s=0.01)
    # print(np.min(np.abs(U)), np.max(np.abs(U)), np.mean(iters), time.time() - starttime)

