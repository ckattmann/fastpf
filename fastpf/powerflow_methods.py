import warnings
import numpy as np
import scipy.sparse
import numba
from numba.pycc import CC
from numba.core.errors import NumbaPerformanceWarning

cc = CC("powerflow_methods_cc")

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

np.seterr("raise")

# YBus Jacobi
# ===========
@cc.export(
    "ybusjacobi", "(complex128[:,:],complex128[:], complex128[:,:], float64, int64)"
)
def ybusjacobi(Y, U, S, eps_s=1.0, max_iters=10000):
    Y = Y.copy()
    S = S.copy()
    diagY = np.diag(Y)
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)
    for i in range(S.shape[0]):
        iters = 0
        s = S[i, :]
        while True:
            IR = Y @ U - np.conj(s / U)
            if np.max(np.abs(U * np.conj(IR))) < eps_s or iters > max_iters:
                break
            U -= IR / diagY
            iters += 1
        u_all[i, :] = U
        iters_all[i] = iters
    return u_all, iters_all


# YBus Gauss-Seidel
# =================
@cc.export(
    "ybusgaussseidel",
    "(complex128[:,:], complex128[:], complex128[:,:], float64, float64, int64)",
)
def ybusgaussseidel(Y, U, S, acc_factor=1.0, eps_s=1, max_iters=5000):
    u_all = np.zeros((S.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0])
    diagY = np.diag(Y)
    for i in range(S.shape[0]):
        s = S[i, :]
        iters = 0
        while True:
            for j in range(Y.shape[0]):
                U[j] -= acc_factor * ((Y[j, :] @ U) - np.conj(s[j] / U[j])) / diagY[j]
            if np.max(np.abs(U * np.conj(Y @ U) - s)) < eps_s or iters > max_iters:
                break
            iters += 1
        u_all[i, :] = U
        iters_all[i] = iters
    return u_all, iters_all


@cc.export(
    "ybusgs_nminus1",
    "(complex128[:,:], int32, int32, complex128, complex128[:], complex128[:], float64, float64, int64)",
)
def gs_nminus1(Y, source_id, target_id, Y_line, s, U, acc_factor, eps_s, max_iters):
    # Switch off one line:
    Y[source_id, target_id] -= Y_line
    Y[target_id, source_id] -= Y_line
    Y[source_id, source_id] += Y_line
    Y[target_id, target_id] += Y_line

    iters = 0
    diagY = np.diag(Y)
    while True:
        for j in range(Y.shape[0]):
            U[j] -= 1.2 * ((Y[j, :] @ U) - np.conj(s[j] / U[j])) / diagY[j]
        if np.max(np.abs(U * np.conj(Y @ U) - s)) < eps_s or iters > max_iters:
            break
        iters += 1
    return U, iters


# YBus Relaxation
# ===============
@cc.export(
    "ybusrelaxation", "(complex128[:,:],complex128[:], complex128[:,:], float64, int64)"
)
def ybusrelaxation(Y, U, S_all, s_eps=0.1, max_iters=20000):
    diagY = np.diag(Y)
    U_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    for i in range(S_all.shape[0]):
        s = S_all[i, :]
        iters = 0
        while True:
            IR = Y @ U - np.conj(s / U)
            SR = np.conj(np.conj(U) * IR)
            max_R_index = np.argmax(np.abs(SR))
            if np.abs(SR[max_R_index]) < s_eps or iters > max_iters:
                break
            iters += 1
            U[max_R_index] -= IR[max_R_index] / diagY[max_R_index]
        U_all[i, :] = U
        iters_all[i] = iters
    return U_all, iters_all


# YBus Newton-Raphson
# ===================
# @cc.export('ybusnewton_old','(complex128[:,:], complex128[:,:], complex128[:], complex128[:,:], int32, float64, int64)')
# def ybusnewton(Y, Yred, U, S_all, slack_index=0, eps_s=1, max_iters=100):

#     numberofnodes = U.shape[0] - 1

#     # This works, but looks stupid:
#     non_slack_indices = np.zeros(numberofnodes, dtype=np.int32)
#     counter = 0
#     for i in range(U.shape[0]):
#         if i != slack_index:
#             non_slack_indices[counter] = i
#             counter += 1

#     S_all = S_all[:,non_slack_indices]

#     U_abs = np.abs(U)[non_slack_indices]
#     # This has to be changed into the given angles:
#     phi = np.zeros(U_abs.size, dtype=np.float64)

#     U_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
#     iters_all = np.zeros(S_all.shape[0])

#     dPQ = np.zeros(2*S_all.shape[1], dtype=np.float64)
#     J = np.zeros((numberofnodes*2, numberofnodes*2), dtype=np.float64)

#     Yredconj = np.conj(Yred)

#     iters = 0

#     for j in range(S_all.shape[0]):
#         iters = 0
#         s = S_all[j,:]

#         while iters <= max_iters:
#             U[non_slack_indices] = U_abs * np.exp(1j*phi)
#             Scalc = U * (np.conj(Y @ U))
#             Smismatch = Scalc[non_slack_indices] - s
#             Pcalc = Smismatch.real
#             Qcalc = Smismatch.imag
#             dPQ = np.hstack((Pcalc, Qcalc))
#             if np.max(np.abs(dPQ)) < eps_s:
#                 break

#             diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
#             diagUred = np.diag(U[non_slack_indices])
#             Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
#             JU = diagIconj + diagUred @ Yredconj

#             J[:numberofnodes, :numberofnodes] = np.real(Jphi)
#             J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
#             J[:numberofnodes, numberofnodes:] = np.real(JU)
#             J[numberofnodes:, numberofnodes:] = np.imag(JU)

#             # dx = np.linalg.inv(-J) @ dPQ
#             dx = np.linalg.solve(-J,dPQ)
#             phi += dx[0:int(dx.shape[0]/2)]
#             U_abs += (dx[int(dx.shape[0]/2):])

#             iters += 1

#         U_all[j] = U
#         iters_all[j] = iters

#     return U_all, iters_all


@cc.export(
    "ybusnewton", "(complex128[:,:], complex128[:,:], complex128[:], float64, int64)"
)
def ybusnewton(Y, S_all, U, eps_s=1, max_iters=100):
    """Assume that node 0 is slack"""

    numberofnodes = U.shape[0] - 1

    S_all = S_all[:, 1:]

    U_abs = np.abs(U)[1:]
    # This has to be changed into the given angles:
    phi = np.zeros(U_abs.size, dtype=np.float64)

    U_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])

    dPQ = np.zeros(2 * S_all.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes * 2, numberofnodes * 2), dtype=np.float64)

    Yredconj = np.conj(Y[1:, 1:])

    iters = 0

    for j in range(S_all.shape[0]):
        iters = 0
        s = S_all[j, :]

        while iters <= max_iters:
            U[1:] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[1:] - s
            Pcalc = Smismatch.real
            Qcalc = Smismatch.imag
            dPQ = np.hstack((Pcalc, Qcalc))
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

            dx = np.linalg.solve(-J, dPQ)
            phi += dx[0 : int(dx.shape[0] / 2)]
            U_abs += dx[int(dx.shape[0] / 2) :]

            iters += 1

        U_all[j] = U
        iters_all[j] = iters

    return U_all, iters_all


@cc.export(
    "ybusnewton_mslacks",
    "(complex128[:,:], complex128[:,:], complex128[:], int64, float64, int64)",
)
def ybusnewton_mslacks(Y, S_all, U, m, eps_s=1, max_iters=100):
    """YBus-Newton Power Flow for multiple slack nodes
    Y       [n x n]     complex128: Admittance matrix, main_slack in first line, other slacks in subsequent lines
    S_all   [l x n]     complex128: Loads in Watt
    U       [n]         complex128: Starting voltages, slack voltages need to be in place
    m                   int64:      Number of additional slacks
    eps_s               float64:    power residual
    max_iters           int64:      number of iterations before break
    """

    numberofnodes = U.shape[0] - 1

    S_all = S_all[:, 1:]

    U_abs = np.abs(U)[1:]
    phi = np.zeros(U_abs.size, dtype=np.float64)

    U_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    additional_slack_voltages = U[1 : m + 1]
    dPQ = np.zeros(2 * S_all.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes * 2, numberofnodes * 2), dtype=np.float64)

    Yredconj = np.conj(Y[1:, 1:])

    iters = 0

    for j in range(S_all.shape[0]):
        iters = 0
        s = S_all[j, :]

        while iters <= max_iters:
            U[1:] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[1:] - s
            dPQ = np.hstack((Smismatch.real, Smismatch.imag))
            for i in range(len(additional_slack_voltages)):
                dPQ[i] = 0
                dPQ[i + numberofnodes] = 0
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
                J[i, :] = np.zeros(numberofnodes * 2)
                J[i, i] = 1
                i_phase = i + numberofnodes
                J[i_phase, :] = np.zeros(numberofnodes * 2)
                J[i_phase, i_phase] = 1

            # Handle generator nodes:
            # for k in range(len(additional_slack_voltages)):
            #     J[i, :] = np.zeros(numberofnodes * 2)
            #     J[i, i] = 1
            #     i_phase = i + numberofnodes
            #     J[i_phase, :] = np.zeros(numberofnodes * 2)
            #     J[i_phase, i_phase] = 1

            dx = np.linalg.solve(-J, dPQ)
            phi += dx[0 : int(dx.shape[0] / 2)]
            U_abs += dx[int(dx.shape[0] / 2) :]

            iters += 1

        U_all[j] = U
        iters_all[j] = iters

    return U_all, iters_all


@cc.export(
    "ybusnewton_full",
    "(complex128[:,:], complex128[:,:], complex128[:], int64, int64, float64, int64)",
)
def ybusnewton_full(Y, S_all, U, n_slack, n_pv, eps_s=1, max_iters=100):
    """YBus-Newton Power Flow for multiple slack nodes
    Y       [n x n]     complex128: Admittance matrix, main_slack in first line, other slacks in subsequent lines
    S_all   [l x n]     complex128: Loads in Watt
    U       [n]         complex128: Starting voltages, slack voltages need to be in place
    n_slack             int64:      Number of additional slacks
    n_pv                int32:      Number of PV nodes, sorted to the end of Y
    # U_pv  [n_pv]      float64:    Fixed voltage magnitude setpoints for generators
    eps_s               float64:    power residual
    max_iters           int64:      number of iterations before break
    """

    numberofnodes = U.shape[0] - 1

    S_all = S_all[:, 1:]

    U_abs = np.abs(U)[1:]
    phi = np.zeros(U_abs.size, dtype=np.float64)

    U_all = np.zeros((S_all.shape[0], U.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0])
    additional_slack_voltages = U[1 : n_slack + 1]
    dPQ = np.zeros(2 * S_all.shape[1], dtype=np.float64)
    J = np.zeros((numberofnodes * 2, numberofnodes * 2), dtype=np.float64)

    Yredconj = np.conj(Y[1:, 1:])

    iters = 0

    for j in range(S_all.shape[0]):
        iters = 0
        s = S_all[j, :]

        while iters <= max_iters:
            U[1:] = U_abs * np.exp(1j * phi)
            Scalc = U * (np.conj(Y @ U))
            Smismatch = Scalc[1:] - s
            dPQ = np.hstack((Smismatch.real, Smismatch.imag))
            for i in range(len(additional_slack_voltages)):
                dPQ[i] = 0
                dPQ[i + numberofnodes] = 0
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
                J[i, :] = np.zeros(numberofnodes * 2)
                J[i, i] = 1
                i_phase = i + numberofnodes
                J[i_phase, :] = np.zeros(numberofnodes * 2)
                J[i_phase, i_phase] = 1

            # Handle generator nodes:
            # for k in range(len(additional_slack_voltages)):
            #     J[i, :] = np.zeros(numberofnodes * 2)
            #     J[i, i] = 1
            #     i_phase = i + numberofnodes
            #     J[i_phase, :] = np.zeros(numberofnodes * 2)
            #     J[i_phase, i_phase] = 1

            dx = np.linalg.solve(-J, dPQ)
            phi += dx[0 : int(dx.shape[0] / 2)]
            U_abs += dx[int(dx.shape[0] / 2) :]

            iters += 1

        U_all[j] = U
        iters_all[j] = iters

    return U_all, iters_all


def ybusnewton_sparse(Y, Yred, U, S, slack_index=0, eps_s=1, max_iters=100):

    numberofnodes = U.shape[0] - 1

    # This looks stupid, but works and is outside the hotpath:
    non_slack_indices = np.zeros(numberofnodes, dtype=np.int32)
    counter = 0
    for i in range(S.shape[1]):
        if i != slack_index:
            non_slack_indices[counter] = i
            counter += 1

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

    # Convert to CSR sparse:
    Y = scipy.sparse.csr_matrix(Y)

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
            if np.max(np.abs(dPQ)) < eps_s:
                break
            iters += 1

            diagIconj = np.diag(np.conj((Y.dot(U))[non_slack_indices]))
            # diagIconj = np.diag(np.conj((Y @ U)[non_slack_indices]))
            diagUred = np.diag(U[non_slack_indices])
            Jphi = 1j * diagUred @ (diagIconj - np.conj(diagUred) @ Yredconj)
            JU = diagIconj + diagUred @ Yredconj

            J[:numberofnodes, :numberofnodes] = np.real(Jphi)
            J[numberofnodes:, :numberofnodes] = np.imag(Jphi)
            J[:numberofnodes, numberofnodes:] = np.real(JU)
            J[numberofnodes:, numberofnodes:] = np.imag(JU)

            J = scipy.sparse.csr_matrix(J)
            # dx = np.linalg.solve(-J,dPQ)
            dx = scipy.sparse.linalg.spsolve(-J, dPQ)
            phi += dx[0 : int(dx.shape[0] / 2)]
            U_abs += dx[int(dx.shape[0] / 2) :]

            J = J.todense()

        u_all[j] = U
        iters_all[j] = iters
    return u_all, iters_all


# Z-Bus Jacobi
# ============
@cc.export(
    "zbusjacobi",
    "(complex128[:,:], complex128[:,:], complex128[:], complex64, float64, int64)",
)
def zbusjacobi(Zred, S_all, U0, slack_voltage, eps_s=1.0, max_iters=20):
    numberofnodes = S_all.shape[1]
    numberofloads = S_all.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes + 1), dtype=np.complex128)
    U_all[:, 0] = slack_voltage
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

    for i in range(numberofloads):
        s = S_all[i, :]
        iters = 0

        # In case of a voltage collapse in the previous computation, U must be reset
        # This is mostly jumped by branch prediction
        if np.abs(U)[0] == 0:
            U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

        while True:
            Ibus = np.conj(s / U)
            U = Zred @ Ibus + slack_voltage
            iters += 1
            if np.max(np.abs(s - U * np.conj(Ibus))) < eps_s:
                break
            # Indicate voltage collapse with U == 0:
            if np.min(np.abs(U)) < 0 or iters > max_iters:
                U = np.zeros_like(U)
                iters = max_iters + 2
                break

        U_all[i, 1:] = U
        iters_all[i] = iters

    return U_all, iters_all


def zbusjacobi_sparse(Zred, S, slack_voltage, eps_s=1.0, max_iters=20):
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    u = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    Zred = scipy.sparse.csr_matrix(Zred)
    for i in range(numberofloads):
        s = S[i, :]
        iters = 0
        while True:
            iters += 1
            Ibus = np.conj(s / u)
            u = Zred.dot(Ibus) + slack_voltage
            if np.max(np.abs(s - u * np.conj(Ibus))) < eps_s or iters > max_iters:
                break
        U_all[i, :] = u
        iters_all[i] = iters

    return U_all, iters_all


@cc.export(
    "zbusjacobi_2slacks",
    "(complex128[:,:], complex128[:,:], complex128[:], complex128, int64, complex128, float64, float64, int64)",
)
def zbusjacobi_2slacks(
    Z,
    S_all,
    u0,
    slack_voltage,
    i_slack,
    U_slack2,
    eps_s=1.0,
    eps_u_slack=0.001,
    max_iters=20,
):
    numberofnodes = S_all.shape[1]
    numberofloads = S_all.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes + 1), dtype=np.complex128)
    U_all[:, 0] = slack_voltage
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

    for i in range(numberofloads):
        S = S_all[i, :]
        iters = 0
        while True:
            I = np.conj(S / U)
            I[0] = 0
            I[0] = (U_slack2 - slack_voltage - Z[0, :] @ I) / Z[0, 0]
            S[0] = U[0] * np.conj(I[0])
            U = Z @ I + slack_voltage
            iters += 1
            if (
                np.max(np.abs(S - U * np.conj(I))) < eps_s
                and np.abs(U[i_slack] - U_slack2) < eps_u_slack
            ) or iters > max_iters:
                break
        U_all[i, 1:] = U
        iters_all[i] = iters

    return U_all, iters_all


@cc.export(
    "zbusjacobi_mslacks",
    "(complex128[:,:], complex128[:,:], complex128[:], complex128, int64, complex128[:], float64, float64, int64)",
)
def zbusjacobi_mslacks(
    Zred,
    S_all,
    u0,
    slack_voltage,
    m,
    U_additional_slacks,
    eps_s=1.0,
    eps_u_slack=0.001,
    max_iters=20,
):
    """Perform the ZBus Jacobi Power Flow in a grid with n>2 slack nodes
    Zred    : [n-1 x n-1], complex128: Inverse of Yred, where the line and column of the main slack node have been deleted
    S_all   : [n-1 x nloads], complex128: Loads in W
    u0      : [n-1], complex128: Starting voltages
    slack_voltage : complex128: Main slack voltage
    m       : int64: number of additional slacks
    U_additional_slacks : [m], complex128: slack voltages of additional slacks
    eps_s   : float64: power residual as convergence condition
    eps_u_slack: float64: voltage residual as convergence condition for additional slacks
    max_iters: int64: number of iterations before assumed divergence and break

    Assumes Zred and S to be ordered, such that
    the nodes representing slack nodes are at the top
    """

    numberofnodes = S_all.shape[1]
    numberofloads = S_all.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes + 1), dtype=np.complex128)
    U_all[:, 0] = slack_voltage
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    U_slack_vector = U.copy()

    for i in range(numberofloads):
        S = S_all[i, :]
        iters = 0
        while True:
            I = np.conj(S / U)
            I[:m] = np.linalg.solve(
                Zred[:m, :m],
                U_additional_slacks - U_slack_vector[:m] - Zred[:m, m:] @ I[m:],
            )
            U = Zred @ I + slack_voltage
            iters += 1
            S[:m] = U[:m] * np.conj(I[:m])
            if (
                np.max(np.abs(S - U * np.conj(I))) < eps_s
                and np.max(np.abs(U[:m] - U_additional_slacks)) < eps_u_slack
            ) or iters > max_iters:
                break
        U_all[i, 1:] = U
        iters_all[i] = iters

    return U_all, iters_all


@cc.export("zbusjacobi_1iter", "(complex128[:,:], complex128[:,:], complex64)")
def zbusjacobi_1iter(Zred, S_all, slack_voltage):
    numberofnodes = S_all.shape[1]
    numberofloads = S_all.shape[0]
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

    for i in range(numberofloads):
        s = S_all[i, :]
        U = Zred @ np.conj(s / U) + slack_voltage
        U_all[i, :] = U

    return U_all


@cc.export("zbusjacobi_2iter", "(complex128[:,:], complex128[:,:], complex64)")
def zbusjacobi_1iter(Zred, S_all, slack_voltage):
    numberofnodes = S_all.shape[1]
    numberofloads = S_all.shape[0]
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

    for i in range(numberofloads):
        s = S_all[i, :]
        U = Zred @ np.conj(s / U) + slack_voltage
        U = Zred @ np.conj(s / U) + slack_voltage
        U_all[i, :] = U

    return U_all


# @cc.export('zbusjacobi','(complex128[:,:], complex128[:,:], complex64, float64, int64)')
# def zbusjacobi(Zred, S_all, slack_voltage, eps_s=1.0, max_iters=20):
#     numberofnodes = S_all.shape[1]
#     numberofloads = S_all.shape[0]
#     iters_all = np.zeros(numberofloads, dtype=np.int32)
#     U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
#     U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

#     for i in range(numberofloads):
#         s = S_all[i,:]
#         iters = 0

#         if np.abs(U)[0] == 0:
#             U = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

#         while True:
#             Ibus = np.conj(s/U)
#             U = Zred @ Ibus + slack_voltage
#             iters += 1
#             if np.max(np.abs(s - U * np.conj(Ibus))) < eps_s:
#                 break
#             if np.min(np.abs(U)) < 0 or iters > max_iters:
#                 U = np.zeros_like(U)
#                 break
#         U_all[i,:] = U
#         iters_all[i] = iters

#     return U_all, iters_all


# Backward/Forward Sweep for Feeders
# ==================================
@cc.export("bfs", "(complex128[:], complex128[:,:], complex64, float64, int64)")
def bfs(Zline, S_all, slack_voltage, eps_s=1.0, max_iters=100):
    S_all = S_all[:, 1:]
    Ubus = np.ones(S_all.shape[1], dtype=np.complex128) * slack_voltage
    Ibus = np.zeros(S_all.shape[1], dtype=np.complex128)
    U_all = np.zeros((S_all.shape[0], Ubus.shape[0]), dtype=np.complex128)
    iters_all = np.zeros(S_all.shape[0], dtype=np.int32)

    for i in range(S_all.shape[0]):
        s = S_all[i, :]
        iters = 0
        while True:
            Ibus = np.conj(s / Ubus)
            Ubus = slack_voltage - np.cumsum(np.cumsum(Ibus[::-1])[::-1] * Zline)
            iters += 1
            if np.max(np.abs(s - Ubus * np.conj(Ibus))) < eps_s or iters > max_iters:
                break
        U_all[i, :] = Ubus
        iters_all[i] = iters

    return U_all, iters_all


def compile(verbose=False):
    cc.verbose = verbose
    cc.compile()


if __name__ == "__main__":
    compile()
