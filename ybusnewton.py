import numpy as np
import numpy.matlib
import json
import matplotlib.pyplot as plt
import time
import numba
import pprint

np.set_printoptions(suppress=True)


def prepdata(grid, linesToRemove=[], baseVoltage=None):
    nodes = grid["nodes"]
    lines = grid["edges"]

    numberofnodes = len(nodes)
    numberoflines = len(lines)

    if not baseVoltage:
        slack_voltages = []
        for n in nodes:
            if "slack_voltage" in n:
                slack_voltages.append(n["slack_voltage"])
        if slack_voltages:
            baseVoltage = np.mean(slack_voltages)
        else:
            print("No Voltages for slack found")
            return

    # Preallocate empty arrays
    u0 = np.ones(numberofnodes, dtype=np.complex128) * baseVoltage
    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)

    # Line Impedances Matrix: Used later for current calculation
    line_impedances = np.zeros((numberofnodes, numberoflines), dtype=np.complex128)

    # Add the connections in 'lines' to the admittancy matrix Y
    for i, line in enumerate(lines):
        if line["id"] not in linesToRemove:

            # Make sure connection goes from lower node number to higher node number,
            # for easy symmetrization with np.transpose later
            if line["source"] > line["target"]:
                from_node, to_node = line["target"], line["source"]
            else:
                from_node, to_node = line["source"], line["target"]

            # Make the connection in the admittancy matrix
            Y[from_node, to_node] += 1 / (line["R"] + 1j * line["X"])

            # Make the connection in the line_impedances matrix
            line_impedances[from_node, i] += 1 / (line["R"] + 1j * line["X"])
            line_impedances[to_node, i] += -1 / (line["R"] + 1j * line["X"])

        else:
            pass
            # print('Line '+line['name']+', number '+str(line['line_number'])+' removed')

    # Symmetrize
    Y = Y + np.transpose(Y)

    # Assign diagonal elements
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])

    # First slack get deleted, next slacks are set to idents
    first_slack = -1
    for i, node in enumerate(nodes):
        if node["is_slack"]:
            if first_slack >= 0:
                print("Multiple slacks still unsupported")
                first_slack = i
            # Y[i,:] = np.zeros_like(Y[i,:])
            # Y[i,i] = 1
            first_slack = i
            slack_voltage = node["slack_voltage"]
            # u0[i] = node['slack_voltage'] * np.exp(1j*node['slack_angle']/360*2*np.pi)
            # u0 = np.delete(u0,i)

    # Calculate Yred and Zred
    Yred = np.delete(Y, first_slack, axis=0)
    Yred = np.delete(Yred, first_slack, axis=1)

    pf_parameters = (Y, Yred, u0, slack_voltages, first_slack)
    extra_parameters = line_impedances

    return pf_parameters, extra_parameters


def pf_plotting(Y, Yred, U, slack_voltage, first_slack_index, S):
    import matplotlib.pyplot as plt

    max_iters = 100
    epsilon = 0.001

    numberofnodes = U.shape[0] - 1
    if first_slack_index == 0:
        non_slack_indices = np.arange(1, numberofnodes + 1)
    elif first_slack_index == numberofnodes:
        non_slack_indices = np.arange(0, numberofnodes)
    else:
        non_slack_indices = np.arange(0, first_slack_index) + np.arange(
            first_slack_index, numberofnodes
        )

    S = S[:, non_slack_indices]

    # Separate Yred into real and imag matrix:
    G = Yred.real
    B = Yred.imag

    U_abs = np.abs(U)[non_slack_indices]

    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dphi = np.zeros((U_abs.size, U_abs.size), dtype=np.float64)

    dPQ = np.zeros(2 * S.shape[1], dtype=np.float64)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j, :]
        old_Us = U.copy()

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < epsilon:
                break
            iters += 1

            # Build Jacobian
            # ==============

            # Build dphi
            for i in range(dphi.shape[0]):
                dphi[i, :] = phi
            dphi = dphi.transpose() - dphi

            # Off-diagonal elements of J
            Uu = np.outer(U_abs, U_abs.transpose())
            Gsin = G * np.sin(dphi)
            Gcos = G * np.cos(dphi)
            Bsin = B * np.sin(dphi)
            Bcos = B * np.cos(dphi)
            H = Uu * (Gsin - Bcos)
            N = Uu * (Gcos + Bsin)
            K = -N
            L = H

            # Diagonal elements of J
            dBU = np.diag(B) * U_abs ** 2
            dGU = np.diag(G) * U_abs ** 2
            for i in range(0, len(H)):
                H[i, i] = -Qcalc[i] - dBU[i]
                N[i, i] = Pcalc[i] + dGU[i]
                K[i, i] = Pcalc[i] - dGU[i]
                L[i, i] = Qcalc[i] - dBU[i]

            # Construct J from H,N,K,L
            Jupper = np.hstack((H, N))
            Jlower = np.hstack((K, L))
            J = np.vstack((Jupper, Jlower))

            # Update U and phi
            dx = np.linalg.inv(-J) @ dPQ
            phi = phi + dx[0 : int(dx.shape[0] / 2)]
            U_abs += (dx[int(dx.shape[0] / 2) :]) * U_abs

            plt.title("Iteration " + str(iters))
            old_Us = np.vstack((old_Us, U))
            colors = plt.cm.Blues(np.linspace(0.1, 1, old_Us.shape[0]))
            for u_index in range(old_Us.shape[0]):
                plt.plot(np.abs(old_Us[u_index, :]), color=colors[u_index])
            plt.grid(True)
            plt.show()
            U_prev = U.copy()

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


@numba.jit(nopython=True, cache=True)
def pf(Y, Yred, U, slack_voltage, first_slack_index, S):
    max_iters = 101
    epsilon = 1

    numberofnodes = U.shape[0] - 1
    if first_slack_index == 0:
        non_slack_indices = np.arange(1, numberofnodes + 1)
    elif first_slack_index == numberofnodes:
        non_slack_indices = np.arange(0, numberofnodes)
    else:
        non_slack_indices = np.arange(0, first_slack_index) + np.arange(
            first_slack_index, numberofnodes
        )

    S = S[:, non_slack_indices]

    # Separate Yred into real and imag matrix:
    G = Yred.real
    B = Yred.imag

    U_abs = np.abs(U)[non_slack_indices]
    U_abs2 = np.abs(U)[non_slack_indices]

    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)
    phi2 = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dphi = np.zeros((U_abs.size, U_abs.size), dtype=np.float64)

    dPQ = np.zeros(2 * S.shape[1], dtype=np.float64)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j, :]

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < epsilon:
                break
            iters += 1

            # Build Jacobian
            # ==============

            # Build dphi
            for i in range(dphi.shape[0]):
                dphi[i, :] = phi
            dphi = dphi.transpose() - dphi

            # Off-diagonal elements of J
            Uu = np.outer(U_abs, U_abs.transpose())
            Gsin = G * np.sin(dphi)
            Gcos = G * np.cos(dphi)
            Bsin = B * np.sin(dphi)
            Bcos = B * np.cos(dphi)
            H = Uu * (Gsin - Bcos)
            N = Uu * (Gcos + Bsin)
            K = -N
            L = H

            # Diagonal elements of J
            dBU = np.diag(B) * U_abs ** 2
            dGU = np.diag(G) * U_abs ** 2
            for i in range(0, len(H)):
                H[i, i] = -Qcalc[i] - dBU[i]
                N[i, i] = Pcalc[i] + dGU[i]
                K[i, i] = Pcalc[i] - dGU[i]
                L[i, i] = Qcalc[i] - dBU[i]

            # Construct J from H,N,K,L
            Jupper = np.hstack((H, N))
            Jlower = np.hstack((K, L))
            J = np.vstack((Jupper, Jlower))
            # Update U and phi
            # ================

            dx = np.linalg.inv(-J) @ dPQ
            # print(dx)
            phi = phi + dx[0 : int(dx.shape[0] / 2)]
            U_abs += (dx[int(dx.shape[0] / 2) :]) * U_abs

            # print('Classic:')
            # print(U_abs)
            # print(phi)

            # print('Alternative:')
            # print('Uabs2:',U_abs2)
            # print('phi2:',phi2)
            # print('-------------')

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


# @numba.jit(nopython=True, cache=True)
def pf2(Y, Yred, U, slack_voltage, first_slack_index, S):
    max_iters = 101
    epsilon = 1

    numberofnodes = U.shape[0] - 1
    if first_slack_index == 0:
        non_slack_indices = np.arange(1, numberofnodes + 1)
    elif first_slack_index == numberofnodes:
        non_slack_indices = np.arange(0, numberofnodes)
    else:
        non_slack_indices = np.arange(0, first_slack_index) + np.arange(
            first_slack_index, numberofnodes
        )

    S = S[:, non_slack_indices]

    U_abs = np.abs(U)[non_slack_indices]

    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dPQ = np.zeros(2 * S.shape[1], dtype=np.float64)

    Yredconj = np.conj(Yred)
    J = np.zeros((numberofnodes, numberofnodes), dtype=np.float64)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j, :]

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < epsilon:
                break
            print(iters, " : ", np.max(np.abs(dPQ)))
            iters += 1

            # Alternative calulation method for J: http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf
            # ==================================================================================================
            diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            # dS / dphi
            Jupper = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            # dS / dU
            Jlower = diagUred @ (diagIconj + diagUred @ Yredconj)

            Jupper_sep = np.hstack((np.real(Jupper), -np.imag(Jupper)))
            Jlower_sep = np.hstack((-np.real(Jlower), np.imag(Jlower)))
            J = np.vstack((Jupper_sep, Jlower_sep))

            # dx = np.linalg.inv(-J) @ dPQ
            dx = np.linalg.solve(-J, dPQ)
            # phi += dx[0:int(dx.shape[0]/2)]
            # U_abs += (dx[int(dx.shape[0]/2):]) * U_abs

            acc_factors = np.arange(0.8, 1.4, 0.02)
            ys = np.zeros_like(acc_factors)

            phi_old = phi.copy()
            U_abs_old = U_abs.copy()
            for i, a in enumerate(acc_factors):
                # a = 1
                print(a, end=" : ")
                print(phi)
                phi += a * dx[0 : int(dx.shape[0] / 2)]
                U_abs += a * (dx[int(dx.shape[0] / 2) :]) * U_abs
                U[non_slack_indices] = U_abs * np.exp(1j * phi)
                Scalc = U * (np.conj(Y @ U))
                Smismatch = Scalc[non_slack_indices] - s
                Pcalc = Smismatch.real
                Qcalc = Smismatch.imag
                dPQ = np.hstack((Pcalc, Qcalc))
                ys[i] = np.max(np.abs(dPQ))
                print(ys[i])
                phi = phi_old.copy()
                U_abs = U_abs_old.copy()
            plt.plot(acc_factors, ys, "x-")
            plt.grid(True)
            plt.show()

            phi += 1 * dx[0 : int(dx.shape[0] / 2)]
            U_abs += 1 * (dx[int(dx.shape[0] / 2) :]) * U_abs

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


# @numba.jit(nopython=True, cache=True)
def pf3(Y, Yred, U, slack_voltage, first_slack_index, S):
    max_iters = 101
    epsilon = 1

    numberofnodes = U.shape[0] - 1
    if first_slack_index == 0:
        non_slack_indices = np.arange(1, numberofnodes + 1)
    elif first_slack_index == numberofnodes:
        non_slack_indices = np.arange(0, numberofnodes)
    else:
        non_slack_indices = np.arange(0, first_slack_index) + np.arange(
            first_slack_index, numberofnodes
        )

    S = S[:, non_slack_indices]

    U_abs = np.abs(U)[non_slack_indices]

    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dPQ = np.zeros(2 * S.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes * 2, numberofnodes * 2), dtype=np.float64)

    Yredconj = np.conj(Yred)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j, :]

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < epsilon:
                break
            iters += 1

            # Alternative calulation method for J: http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf
            # ==================================================================================================
            diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            # dS / dphi:
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            # dS / dU:
            # JU = diagUred @ (diagIconj + diagUred @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            # dx = np.linalg.inv(-J) @ dPQ
            dx = np.linalg.solve(-J, dPQ)
            phi += dx[0 : int(dx.shape[0] / 2)]
            U_abs += dx[int(dx.shape[0] / 2) :]  # * U_abs

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


def pf_verbose(Y, Yred, U, slack_voltage, first_slack_index, S):
    max_iters = 101
    epsilon = 1

    numberofnodes = U.shape[0] - 1
    if first_slack_index == 0:
        non_slack_indices = np.arange(1, numberofnodes + 1)
    elif first_slack_index == numberofnodes:
        non_slack_indices = np.arange(0, numberofnodes)
    else:
        non_slack_indices = np.arange(0, first_slack_index) + np.arange(
            first_slack_index, numberofnodes
        )

    S = S[:, non_slack_indices]

    # Separate Yred into real and imag matrix:
    G = Yred.real
    B = Yred.imag

    U_abs = np.abs(U)[non_slack_indices]
    U_abs2 = np.abs(U)[non_slack_indices]

    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)
    phi2 = np.zeros(U_abs.size, dtype=np.float64)

    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])

    dphi = np.zeros((U_abs.size, U_abs.size), dtype=np.float64)

    dPQ = np.zeros(2 * S.shape[1], dtype=np.float64)

    iters = 0

    for j in range(S.shape[0]):
        iters = 0
        s = S[j, :]

        while iters <= max_iters:
            U[non_slack_indices] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[non_slack_indices] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
            if np.max(np.abs(dPQ)) < epsilon:
                break
            iters += 1

            # Build Jacobian
            # ==============

            # Build dphi
            for i in range(dphi.shape[0]):
                dphi[i, :] = phi
            dphi = dphi.transpose() - dphi

            # Off-diagonal elements of J
            Uu = np.outer(U_abs, U_abs.transpose())
            Gsin = G * np.sin(dphi)
            Gcos = G * np.cos(dphi)
            Bsin = B * np.sin(dphi)
            Bcos = B * np.cos(dphi)
            H = Uu * (Gsin - Bcos)
            N = Uu * (Gcos + Bsin)
            K = -N
            L = H

            # Diagonal elements of J
            dBU = np.diag(B) * U_abs ** 2
            dGU = np.diag(G) * U_abs ** 2
            for i in range(0, len(H)):
                H[i, i] = -Qcalc[i] - dBU[i]
                N[i, i] = Pcalc[i] + dGU[i]
                K[i, i] = Pcalc[i] - dGU[i]
                L[i, i] = Qcalc[i] - dBU[i]

            # Construct J from H,N,K,L
            Jupper = np.hstack((H, N))
            Jlower = np.hstack((K, L))
            J = np.vstack((Jupper, Jlower))

            # Alternative calulation method for J: http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf
            # ==================================================================================================
            I = (Y @ U)[non_slack_indices]
            Ured = U[non_slack_indices]
            # dS / dphi
            # Jupper2 = 1j * np.diag(Ured) @ (np.diag(np.conj(I)) - np.conj(Yred) @ np.diag(np.conj(Ured)))
            Jupper2 = 1j * np.diag(Ured) @ np.conj(np.diag(I) - Yred @ np.diag(Ured))
            # dS / dU
            # Jlower2 = np.diag(Ured) @ (np.diag(np.conj(I)) @ np.diag(np.ones(numberofnodes)) + np.diag(Ured) @ np.conj(Yred @ np.diag(np.ones(numberofnodes))))
            # print('RWL:', dS_dVm)
            # print('CK: ', Jlower2)

            Jupper2_sep = np.hstack((np.real(Jupper2), -np.imag(Jupper2)))
            Jlower2_sep = np.hstack((-np.real(Jlower2), np.imag(Jlower2)))
            J2 = np.vstack((Jupper2_sep, Jlower2_sep))

            dx2 = np.linalg.inv(-J2) @ dPQ
            print(dx2)
            phi2 += dx2[0 : int(dx2.shape[0] / 2)]
            U_abs2 += (dx2[int(dx2.shape[0] / 2) :]) * U_abs2

            print("J :\n", np.round(J))
            # print(np.linalg.inv(-J))

            # print('J2:\n',np.round(J2))
            # print(np.linalg.inv(-J2))

            # print(np.round(J2_2))

            print(np.allclose(J, J2))

            # Update U and phi
            # ================

            dx = np.linalg.inv(-J) @ dPQ
            # print(dx)
            phi = phi + dx[0 : int(dx.shape[0] / 2)]
            U_abs += (dx[int(dx.shape[0] / 2) :]) * U_abs

            # print('Classic:')
            # print(U_abs)
            # print(phi)

            # print('Alternative:')
            # print('Uabs2:',U_abs2)
            # print('phi2:',phi2)
            # print('-------------')

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


if __name__ == "__main__":
    import data.mockgrids as mockgrids
    import data.mockloads as mockloads

    def plotResult(U, iters):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(np.abs(U))
        ax1.set_title("Iteration of U")
        ax1.grid()

        ax2.plot(iters)
        ax2.set_title("Iterations")
        ax2.grid()

        ax3.plot(abs(U.T), "-")
        ax3.set_title("Abs. Voltage")
        ax3.grid()

        ax4.plot(np.angle(U.T) / (2 * np.pi) * 360, "-")
        ax4.set_title("Voltage Angle")
        ax4.grid()

        plt.tight_layout()
        plt.show()

    numberofnodes = 40
    numberofloads = 1

    grid = mockgrids.feeder(numberofnodes)
    print("Grid: Feeder with " + str(numberofnodes) + " nodes")

    S = mockloads.random(grid, maxload=4000 + 1000j, numberofloads=numberofloads)
    print("Loads: random with " + str(numberofloads) + " loads")

    print("-------------------- pf --------------------")
    (Y, Yred, u0, slack_voltage, first_slack), (line_impedances) = prepdata(grid)
    u02 = u0.copy()

    # starttime = time.time()
    # U,i = pf(Y, Yred, u0, slack_voltage, first_slack, S)
    # print('Time  : \t ',time.time() - starttime)
    # print('Mean Iters : \t ',np.mean(i))

    starttime = time.time()
    U2, i2 = pf2(Y, Yred, u02, slack_voltage, first_slack, S)
    print("Time  : \t ", time.time() - starttime)
    print("Mean Iters : \t ", np.mean(i2))

    # starttime = time.time()
    # U3,i3 = pf3(Y, Yred, u02, slack_voltage, first_slack, S)
    # print('Time  : \t ',time.time() - starttime)
    # print('Mean Iters : \t ',np.mean(i))

    # plotResult(U2,i2)

    # print(np.allclose(U,U2))
    # print(np.allclose(U,U3))

    # plt.plot(np.abs(U.T))
    plt.plot(np.abs(U2.T))
    # plt.plot(np.abs(U3.T))
    plt.show()

    #     plt.plot(np.angle(U.T))
    plt.plot(np.angle(U2.T))
    # plt.plot(np.angle(U3.T))
    plt.show()

    # print('-------------------- pf_verbose ------------')
    # Y, Yred, line_impedances, u0, slack_voltage = prepdata(grid)
    # U2,i2 = pf_verbose(Y, Yred, S, u0, slack_voltage)
    # print(U2)
    # plotResult(U2,i2)
