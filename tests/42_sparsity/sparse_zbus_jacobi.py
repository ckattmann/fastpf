import time
import numpy as np
import scipy.sparse
import numba
import matplotlib.pyplot as plt

import powerflow
import powerflow.plotting as plt

# @numba.jit(nopython=True, cache=True)
def zbusjacobi(Zred, S, slack_voltage, eps_s=1.0, max_iters=20):
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    u = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage

    for i in range(numberofloads):
        s = S[i, :]
        iters = 0
        while True:
            iters += 1
            Ibus = np.conj(s / u)
            u = np.dot(Zred, Ibus) + slack_voltage
            if np.max(np.abs(s - u * np.conj(Ibus))) < eps_s or iters > max_iters:
                break
        U_all[i, :] = u
        iters_all[i] = iters

    return U_all, iters_all


def zbusjacobi_sparse_bsr(Zred, S, slack_voltage, eps_s=1.0, max_iters=20):
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    u = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    Zred = scipy.sparse.bsr_matrix(Zred)
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


# @numba.jit(nopython=True, cache=True)
def zbusjacobi_sparse_csr(Zred, S, slack_voltage, eps_s=1.0, max_iters=20):
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


# @numba.jit(nopython=True, cache=True)
def zbusjacobi_sparse_csc(Zred, S, slack_voltage, eps_s=1.0, max_iters=20):
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    u = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    Zred = scipy.sparse.csc_matrix(Zred)
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


def zbusjacobi_sparse_coo(Zred, S, slack_voltage, eps_s=1.0, max_iters=20):
    numberofnodes = S.shape[1]
    numberofloads = S.shape[0]
    iters_all = np.zeros(numberofloads, dtype=np.int32)
    U_all = np.zeros((numberofloads, numberofnodes), dtype=np.complex128)
    u = np.ones(numberofnodes, dtype=np.complex128) * slack_voltage
    Zred = scipy.sparse.csc_matrix(Zred)
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


runtimes_nonsparse = []
runtimes_non_numba = []
runtimes_sparse_bsr = []
runtimes_sparse_csr = []
runtimes_sparse_csc = []
runtimes_sparse_coo = []
nodes = np.arange(10, 201, 10)
# nodes = [10,20,30,40,50,60,70,80,90,100,150,200]

for n in nodes:
    print()
    print(f"{n} nodes:")
    grid = powerflow.mockgrids.meshed(n)
    S = powerflow.mockloads.beta(grid, numberofloads=35040)

    grid_parameters = powerflow.prepdata(grid)
    Yred = grid_parameters["Yred"]
    Zred = np.linalg.inv(grid_parameters["Yred"])
    print(f"Sparsity Yred: {(n**2 - np.count_nonzero(Yred))/n**2*100:.1f}%")
    print(f"Sparsity Zred: {(n**2 - np.count_nonzero(Zred))/n**2*100:.1f}%")
    deleted_nodes = grid_parameters["deleted_node_ids"]
    slack_index = grid_parameters["slack_index"]

    U, iters, runtime = powerflow.zbusjacobi(grid_parameters, S, verbose=False)
    runtimes_nonsparse.append(runtime)
    print(f"Runtime non-sparse: {runtime*1000:.3f} ms")

    S = np.delete(S, deleted_nodes + [slack_index], axis=1)

    starttime = time.time()
    U, iters = zbusjacobi(Zred, S, 400)
    runtimes_non_numba.append(time.time() - starttime)
    print(f"Runtime non-numba: {(time.time() - starttime) * 1000:.3f} ms")

    starttime = time.time()
    U, iters = zbusjacobi_sparse_bsr(Zred, S, 400)
    runtimes_sparse_bsr.append(time.time() - starttime)
    print(f"Runtime sparse BSR: {(time.time() - starttime) * 1000:.3f} ms")

    starttime = time.time()
    U, iters = zbusjacobi_sparse_csr(Zred, S, 400)
    runtimes_sparse_csr.append(time.time() - starttime)
    print(f"Runtime sparse CSR: {(time.time() - starttime) * 1000:.3f} ms")

    starttime = time.time()
    U, iters = zbusjacobi_sparse_csc(Zred, S, 400)
    runtimes_sparse_csc.append(time.time() - starttime)
    print(f"Runtime sparse CSC: {(time.time() - starttime) * 1000:.3f} ms")

    starttime = time.time()
    U, iters = zbusjacobi_sparse_coo(Zred, S, 400)
    runtimes_sparse_coo.append(time.time() - starttime)
    print(f"Runtime sparse COO: {(time.time() - starttime) * 1000:.3f} ms")

fig, ax = plt.subplots()
fig.set_size_inches((4.33, 2.5))

ax.plot(nodes, runtimes_nonsparse, ".-", markersize=6, label="non-sparse")
ax.plot(nodes, runtimes_non_numba, ".-", markersize=6, label="non-numba")
ax.plot(nodes, runtimes_sparse_bsr, ".-", markersize=6, label="BSR")
ax.plot(nodes, runtimes_sparse_csr, ".-", markersize=6, label="CSR")
ax.plot(nodes, runtimes_sparse_csc, ".-", markersize=6, label="CSC")
ax.plot(nodes, runtimes_sparse_coo, ".-", markersize=6, label="COO")
ax.set_xlabel("Number of nodes in grid / -")
ax.set_ylabel("Runtime / s")
ax.legend()
ax.grid()
plt.tight_layout()
plt.save(fig, "sparseZbus2")
plt.show()
