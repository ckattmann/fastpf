import time
import copy
import collections
import numpy as np

from . import validation
from .log import logger


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


def integrate_slacks_into_S(S, grid):
    S_copy = S.copy()
    for i, node in enumerate(grid["nodes"]):
        if node["is_slack"]:
            S_copy[:, i] = (
                np.abs(
                    node["slack_voltage"]
                    * np.exp(1j * node["slack_angle"] / 360 * 2 * np.pi)
                )
                ** 2
            )
    return S_copy


# def extract_U0(grid):
#     number_of_nodes = len(grid["nodes"])
#     U0 = np.zeros((1, number_of_nodes), dtype=np.complex128)
#     for n in grid["nodes"]:
#         u0 = 0 + 0j
#         if "P" in n:
#             S += float(P)
#         if "Q" in n:
#             S += 1j * float(Q)
#     return U0


def extract_S(grid):
    number_of_nodes = len(grid["nodes"])
    S = np.zeros((1, number_of_nodes), dtype=np.complex128)
    for n in grid["nodes"]:
        S[0, n["id"]] = 0 + 0j
        if "P" in n:
            S[0, n["id"]] += float(n["P"])
        if "Q" in n:
            S[0, n["id"]] += 1j * float(n["Q"])
    return S


def collapse_slacks(grid, main_slack_id=None):
    logger.debug("Collapsing identical slack nodes...")

    # Collect Slack Nodes:
    slacknodes = [n for n in grid["nodes"] if n["is_slack"]]
    number_of_slacks = len(slacknodes)

    # If there is only 1 slack, there is nothing to do:
    if number_of_slacks == 1:
        logger.debug(" -> grid has only one slack node, grid unchanged")
        return grid, []

    # If main_slack_id is not explicitly given, its the node with the lowest id:
    if main_slack_id is None:
        main_slack_id = slacknodes[0]["id"]

    # Deep copy before any changes are made:
    grid = copy.deepcopy(grid)
    nodes = grid["nodes"]
    lines = grid["lines"]

    additional_slack_ids = {n["id"] for n in slacknodes} - {main_slack_id}

    deleted_node_ids = []
    for slack1 in [main_slack_id] + list(additional_slack_ids):
        for slack2 in additional_slack_ids:
            if slack2 != slack1 and slack1:
                same_voltages = (
                    nodes[slack1]["slack_voltage"] == nodes[slack2]["slack_voltage"]
                )
                same_angles = (
                    nodes[slack1]["slack_angle"] == nodes[slack2]["slack_angle"]
                )
                if same_voltages and same_voltages:
                    for line in lines:
                        if line["from"] == slack2:
                            line["from"] = slack1
                            logger.debug(
                                f' -> Line {line["id"]} "from"-node: {slack2} -> {slack1}'
                            )
                        if line["to"] == slack2:
                            line["to"] = slack1
                            logger.debug(
                                f' -> Line {line["id"]} "to"-node: {slack2} -> {slack1}'
                            )
                    deleted_node_ids.append(slack2)
                    logger.debug(
                        f" -> Deleting slack node {slack2} and integrating into {slack1}"
                    )
    nodes = [node for node in nodes if node["id"] not in deleted_node_ids]
    grid["nodes"] = nodes

    return grid, deleted_node_ids


def reorder_nodes(grid, S, main_slack_id=None):
    logger.debug(f"Reordering Nodes to bring slacks to top...")

    # Collect slacks:
    slack_nodes = [n for n in grid["nodes"] if n["is_slack"]]
    number_of_slacks = len(slack_nodes)

    # If main_slack_id is not explicitly given, its the node with the lowest id:
    if main_slack_id is None:
        main_slack_id = slack_nodes[0]["id"]
    additional_slack_ids = {n["id"] for n in slack_nodes} - {main_slack_id}

    # 1 slack node in ID 0 -> correct -> return:
    if number_of_slacks == 1 and grid["nodes"][0]["is_slack"] and main_slack_id == 0:
        logger.debug(" -> only slack node is in ID 0, grid unchanged")
        return grid, S, []

    # Check if multiple slacks are correct -> return:
    # slack_ids_correct = True
    # if main_slack_id == 0 and grid["nodes"][0]["is_slack"]:
    #     for slack_node in range(number_of_slacks):
    #         if not grid["nodes"][slack_node]["is_slack"]:
    #             slack_ids_correct = False
    # if slack_ids_correct:
    #     logger.debug(
    #         f" -> {number_of_slacks} slack nodes are in correct order, grid unchanged"
    #     )
    #     return grid, S, []

    # Deep copy before any changes are made:
    grid = copy.deepcopy(grid)
    nodes = grid["nodes"]
    lines = grid["lines"]

    # Sort slack to tops according to this list:
    slack_list = [main_slack_id] + list(additional_slack_ids)

    node_swaps = []
    # Walk through list and swap nodes:
    for i, node_id in enumerate(slack_list):
        if node_id is not i:
            logger.debug(f" -> Swapping nodes {node_id} and {i}")
            for line in lines:
                if line["from"] == i:
                    line["from"] = node_id
                elif line["from"] == node_id:
                    line["from"] = i
                if line["to"] == i:
                    line["to"] = node_id
                elif line["to"] == node_id:
                    line["to"] = i
            for node in nodes:
                if node["id"] == i:
                    node1 = node
                elif node["id"] == node_id:
                    node2 = node
            node1["id"] = node_id
            node2["id"] = i
            S[:, [node_id, i]] = S[:, [i, node_id]]
            node_swaps.append((i, node_id))
    logger.debug(f" -> Slack IDs after reordering: 0 + {additional_slack_ids}")

    nodes.sort(key=lambda n: n["id"])
    grid["nodes"] = nodes

    return grid, S, node_swaps


def process_grid(
    grid,
    S=None,
    main_slack_id=None,
    collapse_identical_slacks=False,
    reorder_slack_nodes=True,
):

    logger.debug("Processing grid...")

    grid = copy.deepcopy(grid)

    if S is None:
        logger.debug(" -> Extracting S from grid dict...")
        S = extract_S(grid)

    # Check Validity of grid and S:
    errors = validation.validate_grid(grid, S)
    if errors:
        raise Exception(f"{len(errors)} errors found in grid during validation")

    # Collect Slack Nodes:
    slacknodes = [n for n in grid["nodes"] if n["is_slack"]]
    numberofslacks = len(slacknodes)

    if numberofslacks == 1:
        main_slack_id = slacknodes[0]["id"]
        additional_slack_ids = {}
    else:
        if not main_slack_id:
            main_slack_id = slacknodes[0]["id"]

    # Delete and reconnect nodes with identical slack voltage:
    if collapse_identical_slacks:
        grid, deleted_node_ids = collapse_slacks(grid, main_slack_id=main_slack_id)
    else:
        deleted_node_ids = []

    # Reorder to bring slacks to top:
    grid, S, node_swaps = reorder_nodes(grid, S, main_slack_id=main_slack_id)

    # At this point, nodes and lines are fixed:
    nodes = grid["nodes"]
    lines = grid["lines"]
    numberofnodes = len(nodes)
    numberoflines = len(lines)
    slacknodes = [n for n in nodes if n["is_slack"]]
    main_slack_id = 0
    main_slack_voltage = nodes[main_slack_id]["slack_voltage"]
    main_slack_angle = nodes[main_slack_id]["slack_angle"]
    additional_slack_ids = {n["id"] for n in slacknodes} - {main_slack_id}

    logger.debug(f"Number of Nodes: {numberofnodes}")
    logger.debug(f"Number of Lines: {numberoflines}")
    logger.debug(f"Number of Slacks: {numberofslacks}")

    # Construct the admittance matrix Y:
    # ==================================

    # Preallocate admittance matrix:
    Y = np.zeros((numberofnodes, numberofnodes), dtype=np.complex128)

    # Add the connections in 'lines' to the admittance matrix Y
    for i, line in enumerate(lines):
        # Make sure connection goes from lower node number to higher node number,
        # for easy symmetrization with np.transpose later:
        if line["from"] > line["to"]:
            from_node, to_node = line["to"], line["from"]
        else:
            from_node, to_node = line["from"], line["to"]

        # Make the connection in the admittancy matrix:
        Y[from_node, to_node] += 1 / (line["R"] + 1j * line["X"])

    # Symmetrize:
    Y = Y + np.transpose(Y)

    # Assign diagonal elements:
    for i in range(0, len(Y)):
        Y[i, i] = -np.sum(Y[i, :])

    # Process shunt impedances:
    # for node in nodes:
    #     i = node["id"]
    #     if "B" not in node:
    #         node["B"] = 0
    #     if "G" not in node:
    #         node["G"] = 0
    #     Y[i, i] = node["G"] + 1j * node["B"]

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

    # Calc Yred:
    Yred = np.delete(Y, main_slack_id, axis=0)
    Yred = np.delete(Yred, main_slack_id, axis=1)

    # Check if Yred is invertible and change line to ident if not:
    # This 'neutralizes' nodes that would be isolated because of a deleted line etc.
    # if np.linalg.det(Yred) == 0:
    #     for n in range(Yred.shape[0]):
    #         if np.sum(np.abs(Yred[n,:])) == 0:
    #             logger.debug(f'Neutralizing Node {n}')
    #             Yred[n,n] = 1
    #             S[:,n] = 0

    # If grid is a feeder, construct Zline for BFS:
    # is_feeder, is_radial = determine_gridshape(grid)
    # if is_feeder:
    #     Z_line = extract_zline(grid)
    # else:
    #     Z_line = None

    grid["nodes"] = nodes
    grid["edges"] = lines

    grid_parameters = {
        "transformed_grid": grid,
        "Y": Y,
        "Yred": Yred,
        "Y_ident": Y_ident,
        "u0": u0,
        "original_slack_indices": [n["id"] for n in slacknodes],
        "slack_index": main_slack_id,
        "main_slack_voltage": nodes[main_slack_id]["slack_voltage"]
        * np.exp(1j * nodes[main_slack_id]["slack_angle"] / 360 * 2 * np.pi),
        "node_ids_to_delete": deleted_node_ids,
        "original_number_of_slacks": len(slacknodes),
        "numslacks": numberofslacks,
        "additional_slack_ids": list(additional_slack_ids),
        "additional_slack_voltages": additional_slack_voltages,
        "node_swaps": node_swaps,
        "S": S,
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
                            if line["from"] == slack2:
                                line["from"] = slack1
                                log(
                                    f'Redirecting line {line["id"]} from node: {slack2} -> {slack1}'
                                )
                            if line["to"] == slack2:
                                line["to"] = slack1
                                log(
                                    f'Redirecting line {line["id"]} to node: {slack2} -> {slack1}'
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
                        if l["from"] == i:
                            l["from"] = original_id
                        if l["to"] == i:
                            l["to"] = original_id
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
                if line["from"] == i:
                    line["from"] = node_id
                elif line["from"] == node_id:
                    line["from"] = i
                if line["to"] == i:
                    line["to"] = node_id
                elif line["to"] == node_id:
                    line["to"] = i
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
            Y[line["to"], line["from"]] += 1 / (line["R"] + 1j * line["X"])
            Y[line["from"], line["to"]] += 1 / (line["R"] + 1j * line["X"])

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
            line_impedances[line["id"], line["from"]] += 1 / (
                line["R"] + 1j * line["X"]
            )
            line_impedances[line["id"], line["to"]] -= 1 / (line["R"] + 1j * line["X"])

    return line_impedances


def integrate_slacks_for_Yident(S, grid):
    S = S.copy()
    for i, node in enumerate(grid["nodes"]):
        if node["is_slack"]:
            U_0 = np.abs(
                node["slack_voltage"]
                * np.exp(1j * node["slack_angle"] / 360 * 2 * np.pi)
            )
            S[:, i] = np.conj(U_0) * U_0
    return S


def buildBusToLineMatrix(grid):
    # Assuming one slack and radial grid

    numberofnodes = len(grid["nodes"])
    numberoflines = len(grid["edges"])

    busToLine = np.zeros((numberoflines, numberofnodes), dtype=np.bool)

    # Prepare an adjacency list of the graph structure
    adj = {node: set() for node in [n["id"] for n in grid["nodes"]]}

    # Step 1: Build Branch-Bus Matrix
    for line in grid["edges"]:
        adj[line["from"]].add((line["id"], line["to"]))
        adj[line["to"]].add((line["id"], line["from"]))

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
