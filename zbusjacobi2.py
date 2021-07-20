import numpy as np
import numba
import time
import collections
import pprint

np.set_printoptions(suppress=True)


def prepdata(grid, linesToRemove=[]):

    nodes = grid["nodes"]
    lines = grid["edges"]
    numberofnodes = len(nodes)
    numberoflines = len(lines)

    # Collect and count slack nodes
    slacknodes = [n["id"] for n in nodes if n["is_slack"]]
    if not slacknodes:
        raise Exception("No slack node found in grid")
    numberofslacknodes = len(slacknodes)

    # Determine the topology of the grid - feeder, radial, or meshed
    if numberofnodes == numberoflines + 1:
        is_radial = True
        sourcenodes = [l["source"] for l in lines]
        sourcenodes_count = collections.Counter(sourcenodes).values()
        if all([c == 1 for c in sourcenodes_count]):
            is_feeder = True
    else:
        is_radial = False
        is_feeder = False

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
            # else:
            #     Y[first_slack,:] += Y[i,:]
            #     Y[:,first_slack] += Y[:,i]
            #     nodes_to_delete.append(i)
    # slack_indices = [first_slack] + nodes_to_delete

    # # Delete old slack nodes from Y and line_impedances
    # Y = np.delete(Y, nodes_to_delete, axis=0)
    # Y = np.delete(Y, nodes_to_delete, axis=1)

    # Assign diagonal elements
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])
    print(Y)

    # Change lines of slacks to idents
    for i, node in enumerate(nodes):
        if node["is_slack"]:
            Y[i, :] = np.zeros_like(Y[i, :])
            Y[i, i] = 1
            u0[i] = node["slack_voltage"] * np.exp(
                1j * node["slack_angle"] / 360 * 2 * np.pi
            )

    # ======================================
    Ycomplete = Y.copy()
    # Change lines of slacks to idents
    for i, node in enumerate(nodes):
        if node["is_slack"]:
            Ycomplete[i, :] = np.zeros_like(Ycomplete[i, :])
            Ycomplete[i, i] = 1
            # u0[i] = node['slack_voltage'] * np.exp(1j*node['slack_angle']/360*2*np.pi)
    print(Ycomplete)
    Z = np.round(np.linalg.inv(Ycomplete))
    print(Z)
    for i, node in enumerate(nodes):
        if node["is_slack"]:
            Z[i, :] = np.zeros_like(Z[i, :])
            Z[i, i] = 1
    print(Z)
    # ======================================

    # Z = 'ColdIsTheVoid'

    # Calculate Yred and Zred
    Yred = np.delete(Y, first_slack, axis=0)
    Yred = np.delete(Yred, first_slack, axis=1)
    Zred = np.linalg.inv(Yred)

    Yhat = 0

    grid_parameters = {
        "Y": Y,
        "Yred": Yred,
        "Yhat": Yhat,
        "Z": Z,
        "Zred": Zred,
        "u0": u0,
        "is_radial": is_radial,
        "is_feeder": is_feeder,
        "number_of_slacks": numberofslacknodes,
        "slacks_have_same_voltage": True,
    }

    return grid_parameters


def pfverbose(Z, U, slack, S):
    epsilon = 1
    U_all = np.zeros((S.shape[0], S.shape[1]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)

    for i in range(S.shape[0]):
        s = S[i, :]
        iters = 0
        print("Start of ZBus PF:")
        print("S :", np.round(s, 2))
        converged = False
        while True:
            iters += 1
            print()
            print("================ Step ", iters, " ================")
            print("U :", np.round(U, 2))
            current = s / np.conj(U)
            print("Current : ", np.round(np.real(current), 2))
            U = np.dot(Z, current)

            print("U after Update : ", np.round(np.real(U), 2))
            R = U * np.conj(current) - s
            print("R after Update : ", np.round(np.real(R), 2))

            if np.max(np.abs(R)) < epsilon:
                print("--> C-C-Convergence")
                break
            if iters >= 5:
                break

        U_all[i, :] = U
        iters_all[i] = iters

    return U_all, iters_all


def pfverbose_gs(Z, U, slack, S):
    epsilon = 1
    U_all = np.zeros((S.shape[0], S.shape[1]), dtype=np.complex128)
    iters_all = np.zeros(S.shape[0], dtype=np.int32)

    for i in range(S.shape[0]):
        s = S[i, :]
        iters = 0
        print("Start of ZBus PF:")
        print("S :", np.round(s, 2))
        converged = False
        while True:
            iters += 1
            print()
            print("================ Step ", iters, " ================")
            print("U :", np.round(U, 2))
            current = s / np.conj(U)
            print("Current : ", np.round(np.real(current), 2))
            for j in reversed(range(Z.shape[0])):
                U[j] = Z[j, :] @ (s / np.conj(U))

            print("U after Update : ", np.round(np.real(U), 2))
            R = U * np.conj(current) - s
            print("R after Update : ", np.round(np.real(R), 2))

            if np.max(np.abs(R)) < epsilon:
                print("--> C-C-Convergence")
                break
            if iters >= 25:
                break

        U_all[i, :] = U
        iters_all[i] = iters

    return U_all, iters_all


if __name__ == "__main__":

    import data.mockgrids as mockgrids
    import data.mockloads as mockloads

    numberofnodes = 50
    grid = mockgrids.feeder(numberofnodes, 1, 0.6, 0)
    # pprint.pprint(grid)
    S = mockloads.fixed(grid, 40, 1)

    grid_parameters = prepdata(grid)
    Z = grid_parameters["Z"]

    slack = 400
    U0 = np.ones(numberofnodes, dtype=np.complex64) * slack
    U, iters = pfverbose_gs(Z, U0, slack, S)
