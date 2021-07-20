import time
import copy
import collections
import numpy as np

import powerflow.plotting as plt


def is_valid(grid, S):

    # Check if all necessary keys are present in the grid dict:
    if not "nodes" in grid:
        return False, 'No "nodes" in grid'
    nodes = grid["nodes"]

    for key in ["id"]:
        for line in grid["nodes"]:
            if key not in line:
                return False, f'{key} missing in grid["edges"] for node {node}'

    if not "edges" in grid:
        return False, 'No "edges" in grid'
    lines = grid["edges"]

    for key in ["id", "source", "target", "R", "X"]:
        for line in grid["edges"]:
            if key not in line:
                return False, f'{key} missing in grid["edges"] for edge {line}'

    # Check Slack Situation:
    slacknodes = [n for n in nodes if n["is_slack"]]
    if not slacknodes:
        return (False, "No slack node found in grid")

    for sn in slacknodes:
        if "slack_voltage" not in sn:
            return (
                False,
                f'is_slack is True for Node {sn["id"]}, but no slack voltage is given',
            )
        if "slack_angle" not in sn:
            return (
                False,
                f'is_slack is True for Node {sn["id"]}, but no slack angle is given',
            )

    # Check if any of the lines lead to a node that does not exist:
    node_ids = [n["id"] for n in nodes]
    for line in lines:
        if line["source"] not in node_ids:
            return (
                False,
                f'Line {line["id"]} has source {line["source"]}, which is not a valid Node ID}}',
            )
        if line["target"] not in node_ids:
            return (
                False,
                f'Line {line["id"]} has target {line["target"]}, which is not a valid Node ID}}',
            )

    # Check if S works with grid:
    if S is not None:
        if type(S) is not np.ndarray:
            return (False, "type(S) is not numpy.ndarray")

        numberofloads, numberofnodes = S.shape
        if numberofnodes != len(nodes):
            return (
                False,
                f"Number of nodes between grid(n={len(nodes)}) and S(n={numberofnodes}) inconsistent",
            )

    return (True, "No Errors found")


def determine_gridshape(grid):
    """Determine the topology of the grid - feeder, radial, or meshed"""

    if len(grid["nodes"]) != len(grid["edges"]) + 1:
        is_radial = False
        is_feeder = False
        return is_feeder, is_radial

    # Check if Feeder: If a source node appears more than once, there is a junction
    sourcenodes = [min(l["source"], l["target"]) for l in grid["edges"]]
    sourcenodes_count = collections.Counter(sourcenodes).values()
    if not all([c == 1 for c in sourcenodes_count]):
        is_feeder = False
        is_radial = True
        return is_feeder, is_radial

    is_feeder = True
    is_radial = True

    return is_feeder, is_radial


def extract_zline(grid):
    # if numberofnodes != numberoflines + 1:
    #     return None

    # if not all([c == 1 for c in sourcenodes_count]):
    #     return None

    sourcenodes = [l["source"] for l in grid["edges"]]
    sourcenodes_count = collections.Counter(sourcenodes).values()

    Zline = np.zeros(len(sourcenodes), dtype=np.complex128)

    for i, line in enumerate(sorted(grid["edges"], key=lambda l: l["source"])):
        Zline[i] = line["R"] + 1j * line["X"]

    return Zline


def calc_grid_parameters(
    grid,
    S,
    slack_treatment="ident",
    main_slack_id=None,
    collapse_identical_slacks=False,
    reorder_slack_nodes=False,
    lines_to_remove=[],
    verbose=False,
):

    if verbose:
        log = print
    else:
        log = lambda *args, **kwargs: None

    log("Calculating grid parameters...")

    grid = copy.deepcopy(grid)

    # Check Validity of grid and S:

    valid, error = is_valid(grid, S)
    if not valid:
        raise Exception(error)

    nodes = grid["nodes"]
    lines = grid["edges"]
    numberofnodes = len(nodes)
    numberoflines = len(lines)

    log(f"Number of Nodes: {numberofnodes}")
    log(f"Number of Lines: {numberoflines}")

    # Collect Slack Nodes:
    slacknodes = [n for n in nodes if n["is_slack"]]
    numberofslacks = len(slacknodes)

    log(f"Number of Slacks: {numberofslacks}")

    if numberofslacks == 1:
        main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {}
    else:
        if not main_slack_id:
            main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {n["id"] for n in slacknodes} - {main_slack_id}

    # Delete nodes with identical slack voltage and reconnect:
    # ========================================================

    node_ids_to_delete = []
    if collapse_identical_slacks:
        for slack1 in [main_slack_id] + list(additional_slack_ids):
            for slack2 in additional_slack_ids:
                if slack2 != slack1 and slack1 not in node_ids_to_delete:
                    same_voltages = (
                        nodes[slack1]["slack_voltage"] == nodes[slack2]["slack_voltage"]
                    )
                    same_angles = (
                        nodes[slack1]["slack_angle"] == nodes[slack2]["slack_angle"]
                    )
                    if same_voltages and same_voltages:
                        for line in lines:
                            if line["source"] == slack2:
                                line["source"] = slack1
                                log(
                                    f'Line {line["id"]} source node: {slack2} -> {slack1}'
                                )
                            if line["target"] == slack2:
                                line["target"] = slack1
                                log(
                                    f'Line {line["id"]} target node: {slack2} -> {slack1}'
                                )
                        node_ids_to_delete.append(slack2)
                        log(
                            f"Deleting slack node {slack2} and integrating into {slack1}"
                        )

        # Deleted nodes from the 'nodes' list:
        nodes = [node for node in nodes if node["id"] not in node_ids_to_delete]

        numberofnodes -= len(node_ids_to_delete)
        slacknodes = [n for n in nodes if n["is_slack"]]
        numberofslacks = len(slacknodes)
        additional_slack_ids -= set(node_ids_to_delete)

    # Reorder to bring slacks to top:
    # ===============================
    # plt.plotgraph(grid, node_labels=True)
    node_swaps = []
    if reorder_slack_nodes:
        new_additional_slack_ids = []
        for i, node_id in enumerate([main_slack_id] + list(additional_slack_ids)):
            if node_id is not i:
                log(f"Swapping nodes {node_id} and {i}")
                for line in lines:
                    if line["source"] == i:
                        line["source"] = node_id
                    elif line["source"] == node_id:
                        line["source"] = i
                    if line["target"] == i:
                        line["target"] = node_id
                    elif line["target"] == node_id:
                        line["target"] = i
                for node in nodes:
                    if node["id"] == i:
                        node1 = node
                    elif node["id"] == node_id:
                        node2 = node
                node1["id"] = node_id
                node2["id"] = i
                S[:, [node_id, i]] = S[:, [i, node_id]]
                node_swaps.append((i, node_id))
            new_additional_slack_ids.append(i)
        main_slack_id = 0
        additional_slack_ids = set(new_additional_slack_ids) - {0}
        log(f"Slack IDs after reordering: 0 + {additional_slack_ids}")
    nodes.sort(key=lambda n: n["id"])

    # At this point, nodes and lines are fixed:
    main_slack_voltage = nodes[main_slack_id]["slack_voltage"]
    main_slack_angle = nodes[main_slack_id]["slack_angle"]

    # plt.plotgraph(grid, node_labels=True)

    # Construct the admittance matrix Y:
    # ==================================

    # Preallocate Arrays
    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)
    line_impedances = np.zeros(
        (numberofnodes, numberoflines), dtype=np.complex128
    )  # Used later for current calculation

    # Add the connections in 'lines' to the admittancy matrix Y
    for i, line in enumerate(lines):
        if line["id"] not in lines_to_remove:
            # Make sure connection goes from lower node number to higher node number,
            # for easy symmetrization with np.transpose later:
            if line["source"] > line["target"]:
                from_node, to_node = line["target"], line["source"]
            else:
                from_node, to_node = line["source"], line["target"]

            # Make the connection in the admittancy matrix:
            Y[from_node, to_node] += 1 / (line["R"] + 1j * line["X"])

            # Make the connection in the line_impedances matrix:
            # line_impedances[from_node, line['id']] += 1 / (line['R'] + 1j * line['X'])
            # line_impedances[to_node, line['id']] += -1 / (line['R'] + 1j * line['X'])

    # Symmetrize:
    Y = Y + np.transpose(Y)

    # Assign diagonal elements:
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])

    # ==================================

    additional_slack_voltages = np.array(
        [
            nodes[i]["slack_voltage"]
            * np.exp(1j * nodes[i]["slack_angle"] / 360 * 2 * np.pi)
            for i in additional_slack_ids
        ],
        dtype=np.complex128,
    )

    # Create Y_ident which has ident line for slack nodes:
    Y_ident = Y.copy()
    all_slack_ids = [main_slack_id] + list(additional_slack_ids)
    Y_ident[all_slack_ids, :] = np.zeros_like(Y_ident[all_slack_ids, :])
    Y_ident[all_slack_ids, all_slack_ids] = 1

    # Prepare u0:
    u0 = np.ones(numberofnodes, dtype=np.complex128) * main_slack_voltage
    u0[main_slack_id] = nodes[main_slack_id]["slack_voltage"] * np.exp(
        1j * nodes[main_slack_id]["slack_angle"] / 360 * 2 * np.pi
    )

    # Calc Yhat:
    Yhat = Y.copy()
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if i != j:
                if Yhat[i, i] != 0:
                    Yhat[i, j] /= Yhat[i, i]
        Yhat[i, i] = 0

    # Calc Yred:
    Yred = np.delete(Y, main_slack_id, axis=0)
    Yred = np.delete(Yred, main_slack_id, axis=1)

    # Check if Yred is invertible and change line to ident if not:
    # This 'neutralizes' nodes that would be isolated because of a deleted line etc.
    # if np.linalg.det(Yred) == 0:
    #     for n in range(Yred.shape[0]):
    #         if np.sum(np.abs(Yred[n,:])) == 0:
    #             log(f'Neutralizing Node {n}')
    #             Yred[n,n] = 1
    #             S[:,n] = 0

    # If grid is a feeder, construct Zline for BFS:
    is_feeder, is_radial = determine_gridshape(grid)
    if is_feeder:
        Z_line = extract_zline(grid)
    else:
        Z_line = None

    grid["nodes"] = nodes
    grid["edges"] = lines

    grid_parameters = {
        "transformed_grid": grid,
        "is_radial": is_radial,
        "is_feeder": is_feeder,
        "Y": Y,
        "Yred": Yred,
        "Yhat": Yhat,
        "Y_ident": Y_ident,
        "Z_lines": line_impedances,
        "Zline": Z_line,
        "u0": u0,
        "original_slack_indices": [n["id"] for n in slacknodes],
        "slack_index": main_slack_id,
        "main_slack_voltage": nodes[main_slack_id]["slack_voltage"]
        * np.exp(1j * nodes[main_slack_id]["slack_angle"] / 360 * 2 * np.pi),
        "node_ids_to_delete": node_ids_to_delete,
        "original_number_of_slacks": len(slacknodes),
        "numslacks": numberofslacks,
        "additional_slack_ids": list(additional_slack_ids),
        "additional_slack_voltages": additional_slack_voltages,
        "node_swaps": node_swaps,
    }

    return grid_parameters


def prepdata(grid, linesToRemove=[], baseVoltage=None, reorder_slack_nodes=False):

    valid, error = is_valid(grid, None)
    if not valid:
        raise Exception(error)

    nodes = grid["nodes"]
    lines = grid["edges"]
    numberofnodes = len(nodes)
    numberoflines = len(lines)

    # Collect Slack Nodes:
    slacknodes = [n for n in nodes if n["is_slack"]]
    slack_voltages = [n["slack_voltage"] for n in slacknodes]
    slack_angles = [n["slack_angle"] for n in slacknodes]

    # Check if slacks have the same voltage:
    slacks_have_same_voltage = np.all(slack_voltages == slack_voltages[0])

    # Determine the base voltage - mean of slack node voltages:
    if not baseVoltage:
        baseVoltage = np.mean(slack_voltages)

    # Construct the admittance matrix Y:
    # ==================================

    # Preallocate Arrays
    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)
    line_impedances = np.zeros(
        (numberofnodes, numberoflines), dtype=np.complex128
    )  # Used later for current calculation

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
            line_impedances[from_node, line["id"]] += 1 / (line["R"] + 1j * line["X"])
            line_impedances[to_node, line["id"]] += -1 / (line["R"] + 1j * line["X"])

    # Symmetrize:
    Y = Y + np.transpose(Y)

    # Delete nodes which can be reduced as identical slack nodes:
    first_slack_id, node_ids_to_delete = attempt_slack_reduction(grid, None)
    for i in node_ids_to_delete:
        # print(first_slack_id,' <- ', i)
        Y[first_slack_id, :] += Y[i, :]
        Y[:, first_slack_id] += Y[:, i]
        numberofnodes -= 1
    Y = np.delete(Y, node_ids_to_delete, axis=0)
    Y = np.delete(Y, node_ids_to_delete, axis=1)

    # Assign diagonal elements:
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])

    # ==================================

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

    # If grid is a feeder, construct Zline for BFS:
    is_feeder, is_radial = determine_gridshape(grid)
    if is_feeder:
        Z_line = extract_zline(grid)
    else:
        Z_line = None

    grid_parameters = {
        "is_radial": is_radial,
        "is_feeder": is_feeder,
        "Y": Y,
        "Yred": Yred,
        "Yhat": Yhat,
        "Y_ident": Y_ident,
        "Zline": Z_line,
        "u0": u0,
        "original_slack_indices": [n["id"] for n in slacknodes],
        "slack_index": first_slack_id,
        "node_ids_to_delete": node_ids_to_delete,
        "number_of_slacks": len(slacknodes),
        "slacks_have_same_voltage": True,
    }

    return grid_parameters


def reorder_grid_nodes(
    grid, S=None, collapse_identical_slacks=False, main_slack_id=None, verbose=0
):

    if verbose > 0:
        log = print
    else:
        log = lambda *args, **kwargs: None

    nodes = grid["nodes"]
    lines = grid["edges"]
    slacknodes = [n for n in nodes if n["is_slack"]]

    if len(slacknodes) == 1:
        main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {}
    else:
        if not main_slack_id:
            main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {n["id"] for n in slacknodes} - {main_slack_id}

    # Delete nodes with identical slack voltage and reconnect:
    # ========================================================
    node_ids_to_delete = []
    if collapse_identical_slacks:
        for slack1 in [main_slack_id] + list(additional_slack_ids):
            for slack2 in additional_slack_ids:
                if slack2 != slack1 and slack1 not in node_ids_to_delete:
                    same_voltages = (
                        nodes[slack1]["slack_voltage"] == nodes[slack2]["slack_voltage"]
                    )
                    same_angles = (
                        nodes[slack1]["slack_angle"] == nodes[slack2]["slack_angle"]
                    )
                    if same_voltages and same_voltages:
                        for line in lines:
                            if line["source"] == slack2:
                                line["source"] = slack1
                                log(
                                    f'Redirecting line {line["id"]} source node: {slack2} -> {slack1}'
                                )
                            if line["target"] == slack2:
                                line["target"] = slack1
                                log(
                                    f'Redirecting line {line["id"]} target node: {slack2} -> {slack1}'
                                )
                        node_ids_to_delete.append(slack2)
                        log(
                            f"Deleting slack node {slack2} and integrating into {slack1}"
                        )

        # Deleted nodes from the 'nodes' list:
        nodes = [node for node in nodes if node["id"] not in node_ids_to_delete]
        S = np.delete(S, node_ids_to_delete, axis=1)

    # Normalize node IDs & fill the gaps:
    nn = []  # normalized nodes
    node_id_dict = {n["id"]: n for n in nodes}
    for i in range(len(nodes)):
        original_id = i
        if i in node_id_dict:
            nn.append(node_id_dict[i])
        else:
            next_node_found = False
            while not next_node_found:
                i += 1
                if i in node_id_dict:
                    print(original_id, i)
                    next_node = node_id_dict[i]
                    del node_id_dict[i]
                    next_node["id"] = original_id
                    nn.append(next_node)
                    for l in lines:
                        if l["source"] == i:
                            l["source"] = original_id
                        if l["target"] == i:
                            l["target"] = original_id
                    next_node_found = True

        # node_id_dict = {n['id']:n for n in nodes}
    print([n["id"] for n in nn])

    nodes = nn
    slacknodes = [n for n in nodes if n["is_slack"]]

    if len(slacknodes) == 1:
        main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {}
    else:
        if not main_slack_id:
            main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {n["id"] for n in slacknodes} - {main_slack_id}

    # Reorder to bring slacks to top:
    # ===============================
    # plt.plotgraph(grid, node_labels=True)
    node_swaps = []
    new_additional_slack_ids = []
    for i, node_id in enumerate([main_slack_id] + list(additional_slack_ids)):
        if node_id is not i:
            log(f"Swapping nodes {node_id} and {i}")
            for line in lines:
                if line["source"] == i:
                    line["source"] = node_id
                elif line["source"] == node_id:
                    line["source"] = i
                if line["target"] == i:
                    line["target"] = node_id
                elif line["target"] == node_id:
                    line["target"] = i
            for node in nodes:
                if node["id"] == i:
                    node1 = node
                elif node["id"] == node_id:
                    node2 = node
            node1["id"] = node_id
            node2["id"] = i
            if S is not None:
                S[:, [node_id, i]] = S[:, [i, node_id]]
            node_swaps.append((i, node_id))
        new_additional_slack_ids.append(i)
    main_slack_id = 0
    additional_slack_ids = set(new_additional_slack_ids) - {0}
    log(f"Slack IDs after reordering: 0 + {additional_slack_ids}")
    nodes.sort(key=lambda n: n["id"])

    grid["nodes"] = nodes
    grid["edges"] = lines

    return grid, node_swaps, S


def calcY(grid, lines_to_remove=[]):
    lines = grid["edges"]
    num_nodes = len(grid["nodes"])
    Y = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    # Add the connections in 'lines' to the admittancy matrix Y
    for i, line in enumerate(lines):
        if line["id"] not in lines_to_remove:
            Y[line["target"], line["source"]] += 1 / (line["R"] + 1j * line["X"])
            Y[line["source"], line["target"]] += 1 / (line["R"] + 1j * line["X"])

    # Assign diagonal elements:
    for i in range(Y.shape[0]):
        Y[i, i] = -np.sum(Y[i, :])

    return Y


def put_slack_voltages_into_u0(grid, base_voltage, u0=None):
    num_nodes = len(grid["nodes"])
    u0 = np.zeros(num_nodes, dtype=np.complex128)
    for node in grid["nodes"]:
        if node["is_slack"]:
            u0[node["id"]] = node["slack_voltage"] * np.exp(
                1j * node["slack_angle"] / 360 * 2 * np.pi
            )
        else:
            u0[node["id"]] = base_voltage
    return u0


def calc_line_impedances(grid, lines_to_remove=[]):

    line_impedances = np.zeros(
        (len(grid["edges"]), len(grid["nodes"])), dtype=np.complex128
    )

    for line in grid["edges"]:
        if line["id"] not in lines_to_remove:
            line_impedances[line["id"], line["source"]] += 1 / (
                line["R"] + 1j * line["X"]
            )
            line_impedances[line["id"], line["target"]] -= 1 / (
                line["R"] + 1j * line["X"]
            )

    return line_impedances


def attempt_slack_reduction(grid, main_slack_id):
    slacknodes = list(filter(lambda n: n["is_slack"], grid["nodes"]))
    if len(slacknodes) == 1:
        return slacknodes[0]["id"], []

    slack_voltages = [n["slack_voltage"] for n in slacknodes]
    slack_angles = [n["slack_angle"] for n in slacknodes]

    # Check if all slack voltages are the same:
    same_voltages = all([u == slack_voltages[0] for u in slack_voltages])
    same_angles = all([phi == slack_angles[0] for phi in slack_angles])
    if same_voltages and same_angles:
        slack_indices = [n["id"] for n in slacknodes]
        if main_slack_id == None:
            main_slack_id = slack_indices[0]
            node_ids_to_delete = slack_indices[1:]
        else:
            node_ids_to_delete = [s for s in slack_indices if s is not main_slack_id]
        first_slack_id = main_slack_id
        return first_slack_id, node_ids_to_delete
    else:
        return slacknodes[0]["id"], []


def integrate_slacks_for_Yident(S, grid):
    for i, node in enumerate(grid["nodes"]):
        if node["is_slack"]:
            U_0 = np.abs(
                node["slack_voltage"]
                * np.exp(1j * node["slack_angle"] / 360 * 2 * np.pi)
            )
            S[:, i] = np.conj(U_0) * U_0
    return S.copy()


def buildBusToLineMatrix(grid):
    # Assuming one slack and radial grid

    numberofnodes = len(grid["nodes"])
    numberoflines = len(grid["edges"])

    busToLine = np.zeros((numberoflines, numberofnodes), dtype=np.bool)

    # Prepare an adjacency list of the graph structure
    adj = {node: set() for node in [n["id"] for n in grid["nodes"]]}

    # Step 1: Build Branch-Bus Matrix
    for line in grid["edges"]:
        adj[line["source"]].add((line["id"], line["target"]))
        adj[line["target"]].add((line["id"], line["source"]))

    paths = [[(None, 0)]]
    while paths:
        path = paths.pop()
        # print('Path:',path)
        nodes_in_path = [p[1] for p in path]
        # print('Nodes in Path:',nodes_in_path)
        possible_paths = [p for p in adj[path[-1][1]] if p[1] not in nodes_in_path]
        for next_path in possible_paths:
            paths.append(path + [next_path])
            for line in tuple(p[0] for p in path if p[0] is not None) + (next_path[0],):
                # print('Current from node',next_path[1],'is flowing though line',line)
                busToLine[line, next_path[1]] = True
        # print('--------------------')

    # print(busToLine.astype(np.int16))
    return busToLine[:, 1:].astype(np.complex128)


def buildZall(grid):
    Zall = np.zeros(len(grid["edges"]), dtype=np.complex128)
    for line in grid["edges"]:
        Zall[line["id"]] = line["R"] + 1j * line["X"]
    return Zall


if __name__ == "__main__":
    import data.mockgrids as mockgrids
    from pprint import pprint as pprint

    grid = mockgrids.feeder(20)
    print(grid)
    prepdata(grid)

    # starttime = time.time()
    # buildBusToLineMatrix(grid)
    # print(time.time() - starttime)
