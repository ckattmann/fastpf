import numpy as np
import numba
import time
import collections

np.set_printoptions(suppress=True)


def prepdata(grid, linesToRemove=[]):
    # print(grid)
    nodes = grid["nodes"]
    numberofnodes = len(nodes)
    lines = grid["edges"]
    numberoflines = len(lines)

    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)

    mean_slack_voltage = np.mean([n["slack_voltage"] for n in nodes if n["is_slack"]])
    u0 = np.ones(numberofnodes, dtype=np.complex128) * mean_slack_voltage

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
    full_Y = Y.copy()

    # Connect slack nodes to first slack node found
    # Slack nodes that can be deleted are first stored in array and deleted later,
    # in order to not upset the numbering in the matrix
    nodes_to_delete = []
    first_slack = -1
    for i, node in enumerate(nodes):
        if node["is_slack"]:
            if first_slack == -1:
                first_slack = i
            else:
                Y[first_slack, :] += Y[i, :]
                Y[:, first_slack] += Y[:, i]
                nodes_to_delete.append(i)
    slack_indices = [first_slack] + nodes_to_delete

    # Delete old slack nodes from Y and line_impedances
    Y = np.delete(Y, nodes_to_delete, axis=0)
    Y = np.delete(Y, nodes_to_delete, axis=1)

    # Assign diagonal elements
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])

    # # ======================================
    # Ycomplete = Y.copy()
    # # Change lines of slacks to idents
    # for i, node in enumerate(nodes):
    #     if node['is_slack']:
    #         Ycomplete[i,:] = np.zeros_like(Ycomplete[i,:])
    #         Ycomplete[i,i] = 1
    #         # u0[i] = node['slack_voltage'] * np.exp(1j*node['slack_angle']/360*2*np.pi)
    # print(Ycomplete)
    # Z = np.round(np.linalg.inv(Ycomplete))
    # for i, node in enumerate(nodes):
    #     if node['is_slack']:
    #         Z[i,:] = np.zeros_like(Z[i,:])
    #         Z[i,i] = 1
    # print(Z)
    # # ======================================
    Z = "ColdIsTheVoid"

    # Calculate Yred and Zred
    Z = np.linalg.inv(Y)
    Yred = np.delete(Y, first_slack, axis=0)
    Yred = np.delete(Yred, first_slack, axis=1)
    Zred = np.linalg.inv(Yred)
    # return Zred, line_impedances, u0

    pf_parameters = (Zred, u0, first_slack, Z)
    extra_parameters = line_impedances

    return pf_parameters, extra_parameters


def attempt_slack_reduction(grid):

    # slacknodes = [n for n in nodes if n['is_slack']]
    slacknodes = list(filter(lambda n: n["is_slack"], grid["nodes"]))

    if len(slacknodes) == 1:
        return slacknodes[0]["id"], [], []

    slack_classes = {}
    for s in slacknodes:
        if (s["slack_voltage"], s["slack_angle"]) in slack_classes:
            slack_classes[s["slack_voltage"], s["slack_angle"]].append(s)
        else:
            slack_classes[s["slack_voltage"], s["slack_angle"]] = [s]

    # slack_voltages = [n['slack_voltage'] for n in slacknodes]
    # slack_angles = [n['slack_angle'] for n in slacknodes]
    # complex_slack_voltages = [(u,phi) for u,phi in zip(slack_voltages, slack_angles)]
    # different_slacks_counter = collections.Counter(complex_slack_voltages)

    # same_voltages = all([u == slack_voltages[0] for u in slack_voltages])
    # same_angles = all([phi == slack_angles[0] for phi in slack_angles])

    if len(slack_classes) == 1:
        slack_indices = [n["id"] for n in slacknodes]
        first_slack_id = slack_indices[0]
        deleted_slack_ids = slack_indices[1:]
        remaining_slack_ids = []
        return first_slack_id, deleted_slack_ids, remaining_slack_ids

    if len(slack_classes) > 1:
        # Find biggest slack_class
        slack_class_sizes = {
            voltage_and_angle: len(nodes)
            for voltage_and_angle, nodes in slack_classes.items()
        }
        # Take [0] of biggest slack_class as main slack
        # Put [1:] of biggest slack_class in slack_nodes_to_delete
        # Put of rest of slack nodes in remaining_slack_nodes
        main_slack = sorted(different_slacks_counter)[0]
        main_slack_indices = [i for i, cv in complex_slack_voltages if c == main_slack]


def prepdata2(grid, linesToRemove=[], baseVoltage=None):

    # valid, error = is_valid(grid)
    # if not valid:
    #     raise Exception(error)

    nodes = grid["nodes"]
    lines = grid["edges"]
    numberofnodes = len(nodes)
    numberoflines = len(lines)

    # Collect slack nodes:
    slacknodes = [n for n in nodes if n["is_slack"]]
    slack_voltages = [n["slack_voltage"] for n in slacknodes]
    slack_angles = [n["slack_angle"] for n in slacknodes]

    # Determine the base voltage - mean of slack node voltages:
    if not baseVoltage:
        baseVoltage = np.mean(slack_voltages)

    # Preallocate empty arrays
    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)
    line_impedances = np.zeros(
        (numberofnodes, numberoflines), dtype=np.complex128
    )  # Used later for current calculation

    first_slack_id, deleted_node_ids = attempt_slack_reduction(grid)

    # Construct the admittance matrix Y:
    # ==================================

    # Add the connections in 'lines' to the admittancy matrix Y
    for i, line in enumerate(lines):
        if line["id"] not in linesToRemove:
            # Make sure connection goes from lower node number to higher node number,
            # for easy symmetrization with np.transpose later:
            if line["source"] > line["target"]:
                from_node, to_node = line["target"], line["source"]
            else:
                from_node, to_node = line["source"], line["target"]

            # Make the connection in the admittancy matrix:
            Y[from_node, to_node] += 1 / (line["R"] + 1j * line["X"])

            # Make the connection in the line_impedances matrix:
            line_impedances[from_node, i] += 1 / (line["R"] + 1j * line["X"])
            line_impedances[to_node, i] += -1 / (line["R"] + 1j * line["X"])

    # Symmetrize:
    Y = Y + np.transpose(Y)

    # Delete nodes which can be reduced as identical slack nodes:
    for i in deleted_node_ids:
        # print(first_slack_id,' <- ', i)
        Y[first_slack_id, :] += Y[i, :]
        Y[:, first_slack_id] += Y[:, i]
        numberofnodes -= 1
    Y = np.delete(Y, deleted_node_ids, axis=0)
    Y = np.delete(Y, deleted_node_ids, axis=1)

    # Assign diagonal elements:
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])

    # Create Y_ident which has ident line for slack nodes:
    Y_ident = Y.copy()
    Y_ident[first_slack_id, :] = np.zeros_like(Y_ident[first_slack_id, :])
    Y_ident[first_slack_id, first_slack_id] = 1

    # Prepare u0:
    u0 = np.ones(numberofnodes, dtype=np.complex128) * baseVoltage
    u0[first_slack_id] = nodes[first_slack_id]["slack_voltage"] * np.exp(
        1j * nodes[first_slack_id]["slack_angle"] / 360 * 2 * np.pi
    )

    # Calc Yhat:
    Yhat = Y.copy()
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if i != j:
                Yhat[i, j] /= Yhat[i, i]
        Yhat[i, i] = 0

    # Calc Yred:
    Yred = np.delete(Y, first_slack_id, axis=0)
    Yred = np.delete(Yred, first_slack_id, axis=1)

    grid_parameters = {
        "Y": Y,
        "Yred": Yred,
        "Yhat": Yhat,
        "Y_ident": Y_ident,
        # 'Zline': Zline,
        "u0": u0,
        # 'is_radial': is_radial,
        # 'is_feeder': is_feeder,
        "original_slack_indices": [n["id"] for n in slacknodes],
        "slack_index": first_slack_id,
        "deleted_node_ids": deleted_node_ids,
        "number_of_slacks": len(slacknodes),
        "remaining_slack_nodes": remaining_slack_nodes,
        "slacks_have_same_voltage": True,
    }

    return grid_parameters


def pf_verbose2(Zred, U, slack, S):
    epsilon = 0.01
    U_all = np.zeros((S.shape[0], S.shape[1]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)

    for i in range(S.shape[0]):
        s = S[i, :]
        iters = 0
        print("S :", np.round(s, 2))
        converged = False
        while True:
            print()
            print("================ Step ", iters, " ================")
            print("U :", np.round(U, 2))
            current = s / np.conj(U)
            print("Current : ", np.round(current, 2))
            U = np.dot(Zred, current) + slack
            iters += 1

            print("U after Update : ", np.round(U, 2))
            R = U * np.conj(current) - s
            print("R after Update : ", np.round(R, 2))

            if np.max(np.abs(R)) < epsilon:
                print("--> C-C-Convergence")
                break
            if iters >= 10:
                break

        U_all[:, i] = U
        iters_all[i] = iters

    return U_all, iters_all


def pf_verbose3(Zred, U, S):
    S = S[:, 1:]
    LoadP = S.T.real
    LoadQ = S.T.imag
    slack = np.mean(U)
    epsilon = 0.0001
    numberOfNodes = LoadP.shape[0]
    numberOfLoads = LoadP.shape[1]
    all_iters = np.zeros(numberOfLoads, dtype=np.int32)
    all_voltages = np.zeros((numberOfNodes, numberOfLoads), dtype=np.complex128)
    u = np.ones(numberOfNodes, dtype=np.complex128) * slack

    S_ref_complete_conj = np.conj(LoadP + 1j * LoadQ)

    for x in range(numberOfLoads):
        iters = 0
        S_ref = S_ref_complete_conj[:, x]

        converged = False
        while True:
            print()
            print("================ Step ", iters, " ================")
            print("U :", np.round(u, 2))
            current = S_ref / np.conj(u)
            u = np.dot(Zred, current) + slack
            S = u * np.conj(current)

            for i, s in enumerate(S):
                dP = np.abs(s.real - LoadP[i, x])
                dQ = np.abs(s.imag - LoadQ[i, x])
                if dP > epsilon or dQ > epsilon:
                    converged = False
                    break
                converged = True
            if converged or iters >= 200:
                break
            iters += 1

        all_voltages[:, x] = u
        all_iters[x] = iters

    return all_voltages, all_iters


# @numba.jit(nopython=True, cache=True)
def pf_verbose(Zred, U, S):
    LoadP = S.T.real
    LoadQ = S.T.imag
    slack = np.mean(U)
    epsilon = 0.0001
    numberOfNodes = LoadP.shape[0]
    numberOfLoads = LoadP.shape[1]
    all_iters = np.zeros(numberOfLoads, dtype=np.int32)
    all_voltages = np.zeros((numberOfNodes, numberOfLoads), dtype=np.complex128)
    u = np.ones(numberOfNodes, dtype=np.complex128) * slack

    S_ref_complete_conj = np.conj(LoadP + 1j * LoadQ)

    for x in range(numberOfLoads):
        iters = 0
        S_ref = S_ref_complete_conj[:, x]

        converged = False
        while True:
            current = S_ref / np.conj(u)
            u = np.dot(Zred, current) + slack
            S = u * np.conj(current)

            for i, s in enumerate(S):
                dP = np.abs(s.real - LoadP[i, x])
                dQ = np.abs(s.imag - LoadQ[i, x])
                if dP > epsilon or dQ > epsilon:
                    converged = False
                    break
                converged = True
            if converged or iters >= 200:
                break
            iters += 1

        all_voltages[:, x] = u
        all_iters[x] = iters

    return all_voltages, all_iters


@numba.jit(nopython=True, cache=True)
def pf_original(Zred, S, slack):
    epsilon = 0.01
    numberOfNodes = S.shape[1] - 1
    numberOfLoads = S.shape[0]
    iters = np.zeros(numberOfLoads, dtype=np.int32)
    all_voltages = np.zeros((numberOfNodes, numberOfLoads), dtype=np.complex128)
    u = np.ones(numberOfNodes, dtype=np.complex128) * slack

    for x in range(numberOfLoads):
        iterations = 0
        s = S[x, 1:]

        while True:
            iterations += 1
            current = np.conj(s / u)
            u = np.dot(Zred, current) + slack
            R = s - u * np.conj(current)
            if np.max(np.abs(R)) < epsilon:
                break
            if iterations > 200:
                break

        all_voltages[:, x] = u
        iters[x] = iterations

    return all_voltages, iters


@numba.jit(nopython=True, cache=True)
def pf_original2(Zred, S, slack, eps_s=0.01, max_iters=200):
    numberOfNodes = S.shape[1]
    numberOfLoads = S.shape[0]
    iters = np.zeros(numberOfLoads, dtype=np.int32)
    all_voltages = np.zeros((numberOfNodes, numberOfLoads), dtype=np.complex128)
    U = np.ones(numberOfNodes, dtype=np.complex128) * slack

    for i in range(numberOfLoads):
        iterations = 0
        s = S[i, :]
        while True:
            iterations += 1
            I = np.conj(s / U)
            U = Zred @ I + slack
            R = s - U * np.conj(I)
            if np.max(np.abs(R)) < eps_s:
                break
            if iterations > 200:
                break

        all_voltages[:, i] = U
        iters[i] = iterations

    return all_voltages, iters


def pf(Zred, U, first_slack_index, S):
    numberofnodes = U.shape[0] - 1
    if first_slack_index == 0:
        non_slack_indices = np.arange(1, numberofnodes + 1)
    elif first_slack_index == numberofnodes:
        non_slack_indices = np.arange(0, numberofnodes)
    else:
        non_slack_indices = np.hstack(
            (
                np.arange(0, first_slack_index),
                np.arange(first_slack_index, numberofnodes),
            )
        )

    S = S[:, non_slack_indices]
    slack = 400
    u_all, iter_all = pf_original(Zred, S, slack)
    return u_all, iter_all


# @numba.jit(nopython=True, cache=True)
def pf_slack_mod(Zred, Yred, S, slack, quiet_slack_node_indices):
    epsilon = 0.01
    numberOfNodes = S.shape[1] - 1
    numberOfLoads = S.shape[0]
    iters = np.zeros(numberOfLoads, dtype=np.int32)
    all_voltages = np.zeros((numberOfNodes, numberOfLoads), dtype=np.complex128)
    u = np.ones(numberOfNodes, dtype=np.complex128) * slack

    for x in range(numberOfLoads):
        iterations = 0
        s = S[x, 1:]

        while True:
            iterations += 1
            current = np.conj(s / u)
            u = np.dot(Zred, current) + slack
            R = s - u * np.conj(current)
            if np.max(np.abs(R)) < epsilon:
                break
            if iterations > 200:
                break

        all_voltages[:, x] = u
        iters[x] = iterations

    return all_voltages, iters


if __name__ == "__main__":

    import data.mockgrids as mockgrids
    import data.mockloads as mockloads

    grid = mockgrids.feeder(3, numberofslacks=2)
    S = mockloads.fixed(grid, 100)

    print(grid)
    print(S)

    # numberOfNodes = 100
    # numberOfLoads = 35040 * 1

    # print('Solving '+str(numberOfLoads)+' load flows in a grid with '+str(numberOfNodes)+' nodes')

    # grid = mockgrids.feeder(numberOfNodes)
    # S = mockloads.random(grid, maxload=1000, numberofloads=numberOfLoads)

    grid_parameters = prepdata2(grid)
    Zred = grid_parameters["Zred"]
    Yred = grid_parameters["Yred"]
    U, iters = pf_slack_mod(Zred, Yred, S, slack, quiet_slack_node_indices)

    # starttime = time.time()
    # U, iters = pf(*pf_parameters, S)
    # print()
    # print('Time/s : \t ',time.time()-starttime)
    # print('Mean iters : \t ',np.mean(iters))
    # print('Max iters : \t ',np.max(iters))
    # # print('minU  : ', np.min(U, axis=0))
