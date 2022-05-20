import queue
import numpy as np

# import powerflow


def find_impedance_paths(grid):
    """Find impedance from slack to every node in a radial grid
    Parameters:
        grid: [dict] grid data structure containing nodes and edges
    Returns:
        impedance_to_nodes: [dict] <node_name>: [complex] Impedance in Ohms
        path_to_nodes: [dict] <node_name>: [list] of node_names in the path from slack to node
    """

    nodes = grid["nodes"]
    visited = {node["name"]: False for node in nodes}

    impedance_to_nodes = {node["name"]: 0 for node in nodes}
    path_to_nodes = {node["name"]: [] for node in nodes}

    # Use node names, because the ids change during the reduction
    node_id_to_node_name = {n["id"]: n["name"] for n in nodes}

    # Build a dict which contains the neighbours for every node:
    adjacency_dict = {node["name"]: [] for node in nodes}
    for line in grid["edges"]:
        target_name = node_id_to_node_name[line["target"]]
        source_name = node_id_to_node_name[line["source"]]
        adjacency_dict[source_name].append((target_name, line["R"] + 1j * line["X"]))
        adjacency_dict[target_name].append((source_name, line["R"] + 1j * line["X"]))

    # Find Slack node:
    slack_node_names = [node["name"] for node in nodes if node["is_slack"]]
    if len(slack_node_names) == 1:
        slack_node_name = slack_node_names[0]
    else:
        print("0 or 2 or more slack nodes!")
        return None

    # Find adjacent nodes and fill impedances_of_paths using BFS:
    # ===========================================================

    q = queue.Queue()
    q.put((slack_node_name, 0, 0, []))

    while q.qsize():
        node_name, total_path_impedance, line_impedance, node_path = q.get()
        if visited[node_name]:
            continue
        impedance_to_node = total_path_impedance + line_impedance
        impedance_to_nodes[node_name] = impedance_to_node
        path_to_nodes[node_name] = node_path + [node_name]
        # print(f'{node_name}: {round(impedance_to_node.real*1000,1)} mOhm, Path: {" -> ".join([str(i) for i in node_path])}')
        visited[node_name] = True
        for next_node_name, line_impedance in adjacency_dict[node_name]:
            # print(f'\t{next_node_name}: {round(line_impedance.real*1000,1)} mOhm {"-> visited" if visited[next_node_name] else ""}')
            if visited[next_node_name]:
                continue
            q.put(
                (
                    next_node_name,
                    impedance_to_node,
                    line_impedance,
                    node_path + [node_name],
                )
            )

    # print(impedance_to_nodes)

    return impedance_to_nodes, path_to_nodes


def find_candidates_end(grid, S, U0=400 * 0.9, excluded_node_names=[]):
    nodes = grid["nodes"]
    lines = grid["edges"]
    candidates = {}

    for node in nodes:
        node_id = node["id"]

        # Check excluded_nodes
        # for reduction to a single feeder,
        # the nodes in the feeder with the highest impedance path must not be reduced:
        if "name" in node and node["name"] in excluded_node_names:
            continue

        # Shortcut continue for slacks, obviously not removable:
        if node["is_slack"]:
            continue

        # Find connected lines and nodes:
        connected_nodes = []
        connected_lines = []
        for line_i, line in enumerate(lines):
            if node_id == line["source"]:
                connected_nodes.append(line["target"])
                connected_lines.append(line)
            elif node_id == line["target"]:
                connected_nodes.append(line["source"])
                connected_lines.append(line)

        # End Node -> Delete Node and Line:
        if len(connected_nodes) == 1 and connected_nodes[0]:
            candidates[node_id] = {}
            candidates[node_id]["node"] = node
            candidates[node_id]["connected_node_ids"] = connected_nodes
            candidates[node_id]["connected_lines"] = connected_lines
            maxS = max(abs(S[:, node_id]))
            Z = connected_lines[0]["R"] + 1j * connected_lines[0]["X"]
            Uexp = Z * np.conj(maxS / U0)
            # print(f'Node: {node_id}, R={round(Z.real,6)}, X={round(Z.imag,6)}, maxS({node_id})={round(maxS)} -> Uexp={round(abs(Uexp),4)} V')
            candidates[node_id]["Uexp"] = Uexp
    return candidates


def reduce_one_node_end(grid, S, U0=400 * 0.9, excluded_node_names=[]):

    candidates = find_candidates_end(
        grid, S, U0=U0, excluded_node_names=excluded_node_names
    )
    if not candidates:
        return grid, S, {}

    best_candidate = list(sorted(candidates.values(), key=lambda c: abs(c["Uexp"])))[0]
    # print(f'Expected voltage error: {best_candidate["Uexp"]}')

    nodes = grid["nodes"]
    lines = grid["edges"]
    node = best_candidate["node"]

    parent_node_id = best_candidate["connected_node_ids"][0]
    line = best_candidate["connected_lines"][0]

    node_reduced = best_candidate["node"]
    nodes.remove(node_reduced)
    lines.remove(line)

    grid = {"nodes": nodes, "edges": lines, "grid": grid["grid"]}

    S[:, parent_node_id] += (
        S[:, node["id"]]
        + (line["R"] + 1j * line["X"]) * np.conj(abs(S[:, node["id"]]) / abs(U0)) ** 2
    )
    grid, S = powerflow.grid_reduction.lossless.normalise_node_ids(grid, S)
    S = S[:, : len(nodes)]

    return grid, S, node_reduced


def find_candidates_transit(grid, S, excluded_node_names=[]):

    nodes = grid["nodes"]
    lines = grid["edges"]
    candidates = {}

    for node in nodes:
        node_id = node["id"]

        # Check excluded_nodes
        # for reduction to a single feeder,
        # the nodes in the feeder with the highest impedance path must not be reduced:
        if "name" in node and node["name"] in excluded_node_names:
            continue

        # Shortcut continue for slacks, obviously not removable:
        if node["is_slack"]:
            continue

        # Find connected lines and nodes:
        connected_nodes = []
        connected_lines = []
        for line_i, line in enumerate(lines):
            if node_id == line["source"]:
                connected_nodes.append(line["target"])
                connected_lines.append(line)
            elif node_id == line["target"]:
                connected_nodes.append(line["source"])
                connected_lines.append(line)

        if len(connected_nodes) == 2 and connected_nodes[0]:
            candidates[node_id] = {}
            candidates[node_id]["node"] = node
            candidates[node_id]["connected_node_ids"] = connected_nodes
            candidates[node_id]["connected_lines"] = connected_lines
            candidates[node_id]["maxS"] = max(abs(S[:, node_id]))

    return candidates


def reduce_one_node_transit(grid, S, excluded_node_names=[]):
    candidates = find_candidates_transit(
        grid, S, excluded_node_names=excluded_node_names
    )
    if not candidates:
        return grid, S, {}

    nodes = grid["nodes"]
    lines = grid["edges"]

    best_candidate = list(sorted(candidates.values(), key=lambda c: abs(c["maxS"])))[0]

    # Create new line 13:
    connected_lines = best_candidate["connected_lines"]
    new_line = {}
    new_line["id"] = connected_lines[0]["id"]
    new_line["source"] = best_candidate["connected_node_ids"][0]
    new_line["target"] = best_candidate["connected_node_ids"][1]
    new_line["R"] = connected_lines[0]["R"] + connected_lines[1]["R"]
    new_line["X"] = connected_lines[0]["X"] + connected_lines[1]["X"]
    if "length_m" in connected_lines[0] and "length_m" in connected_lines[1]:
        new_line["length_m"] = (
            connected_lines[0]["length_m"] + connected_lines[1]["length_m"]
        )

    # Change S at nodes 1 and 3:
    node1_id = best_candidate["connected_node_ids"][0]
    node2_id = best_candidate["node"]["id"]
    node3_id = best_candidate["connected_node_ids"][1]
    if (
        best_candidate["connected_lines"][0]["source"] == node1_id
        or best_candidate["connected_lines"][0]["target"] == node1_id
    ):

        line12 = best_candidate["connected_lines"][0]
        line23 = best_candidate["connected_lines"][1]
    else:
        line12 = best_candidate["connected_lines"][1]
        line23 = best_candidate["connected_lines"][0]

    Z12 = line12["R"] + 1j * line12["X"]
    Z23 = line23["R"] + 1j * line23["X"]

    S[:, node1_id] += Z23 / (Z12 + Z23) * S[:, node2_id]
    S[:, node3_id] += Z12 / (Z12 + Z23) * S[:, node2_id]

    # Delete Node 2 and lines 12 and 23 and append new line:
    node_reduced = best_candidate["node"]
    nodes.remove(best_candidate["node"])
    for line in connected_lines:
        lines.remove(line)
    lines.append(new_line)
    grid = {"nodes": nodes, "edges": lines, "grid": grid["grid"]}

    grid, S = powerflow.grid_reduction.lossless.normalise_node_ids(grid, S)

    S = S[:, : len(grid["nodes"])]

    return grid, S, node_reduced


def reduce_to_n_nodes(grid, S, n):

    nodes_reduced = []
    while len(grid["nodes"]) > n:

        # First, try intermediate reduction, then end reduction
        grid, S, node_reduced = reduce_one_node_transit(grid, S)
        if node_reduced:
            # print(f'Reducing node {node_reduced["id"]} with transit reduction')
            nodes_reduced.append(node_reduced)
            continue
        else:
            grid, S, node_reduced = reduce_one_node_end(grid, S)
            if not node_reduced:
                break
            # print(f'Reducing node {node_reduced["id"]} with end reduction')
            nodes_reduced.append(node_reduced)

    return grid, S, nodes_reduced


def reduce_to_single_feeder(grid, S):

    impedance_of_paths, path_to_nodes = find_impedance_paths(grid)
    largest_impedance_path = sorted(
        impedance_of_paths.items(), key=lambda i: abs(i[1])
    )[-1]
    final_node_name, impedance_to_slack = largest_impedance_path

    path_of_largest_impedance = path_to_nodes[final_node_name]

    nodes_reduced = []

    # First, try intermediate reduction, then end reduction
    while True:
        grid, S, node_reduced = reduce_one_node_transit(
            grid, S, excluded_node_names=path_of_largest_impedance
        )
        if node_reduced:
            nodes_reduced.append(node_reduced)
            continue
        else:
            grid, S, node_reduced = reduce_one_node_end(
                grid, S, excluded_node_names=path_of_largest_impedance
            )
            if not node_reduced:
                break
            nodes_reduced.append(node_reduced)

    return grid, S, nodes_reduced


def plot_voltage_comparison(U, U_org, filename=None):
    # Plot a comparison of minimal voltage at all 1440 times:
    import matplotlib.ticker as ticker

    minU = np.min(abs(U), axis=1)
    minUorg = np.min(abs(U_org), axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.setsize(fig, size=1.5)
    ax1.plot(minU)
    ax1.plot(minUorg)
    ax1.set_ylabel("Min. abs. Voltage / V")
    ax1.set_xlabel("Time of day / hh:mm")
    ax1.xaxis.set_major_locator(
        ticker.FixedLocator([t * 60 for t in (0, 6, 12, 18, 24)])
    )
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x//60}:00"))
    ax1.grid()
    ax2.plot(abs(minU - minUorg))
    ax2.set_ylabel("Difference / V")
    ax2.set_xlabel("Time of day / hh:mm")
    ax2.xaxis.set_major_locator(
        ticker.FixedLocator([t * 60 for t in (0, 6, 12, 18, 24)])
    )
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x//60}:00"))
    ax2.grid()
    plt.tight_layout()
    if filename:
        plt.save(fig, filename + ".eps")
    plt.show()


def plotgraph_highlight_path(
    grid,
    filename="",
    S=None,
    ax=None,
    verbose=False,
    nodes_in_path=[],
    **additional_args,
):
    import copy
    import networkx as nx
    import powerflow.plotting as plt

    grid = copy.deepcopy(grid)
    g = nx.Graph()
    nodes = grid["nodes"]
    node_ids = [g["id"] for g in nodes]
    if verbose:
        print(f"Plotting {len(node_ids)} Nodes...")
    positions = {n["id"]: (n["x"], n["y"]) for n in nodes}
    g.add_nodes_from(node_ids)
    for e in grid["edges"]:
        if e["source"] in nodes_in_path and e["target"] in nodes_in_path:
            color = "darkslategray"
        else:
            color = "lightgray"
        g.add_edge(e["source"], e["target"], color=color)

    if S is not None:
        for i, n in enumerate(nodes):
            if np.sum(S[:, n["id"]]) != 0:
                nodes[i]["noload"] = False
            else:
                nodes[i]["noload"] = True

    # Set Slacks to green, loads to orange, rest to gray:
    node_color = [
        "darkslategray" if n["id"] in nodes_in_path else "lightgray"
        for n in grid["nodes"]
    ]
    node_color = [
        "orangered" if not n["noload"] else nc
        for n, nc in zip(grid["nodes"], node_color)
    ]
    node_color = [
        "green" if n["is_slack"] else nc for n, nc in zip(grid["nodes"], node_color)
    ]

    node_size = [18 if not n["noload"] else 3 for n in grid["nodes"]]
    node_size = [18 if n["is_slack"] else ns for n, ns in zip(grid["nodes"], node_size)]

    edge_colors = [g[i][j]["color"] for i, j in g.edges]

    if ax == None:
        fig, ax = plt.subplots()
        plt.setsize(fig, 1)
        ax.axis("off")
        plt.margins(0.01)
        ax.margins(0.02)
        showplot = True
    else:
        showplot = False

    nx.draw_networkx(
        g,
        ax=ax,
        pos=positions,
        node_color=node_color,
        node_size=node_size,
        edge_color=edge_colors,
        width=1.5,
        with_labels=False,
        **additional_args,
    )

    plt.tight_layout()

    if showplot:
        if filename:
            plt.save(fig, f"{filename}.eps")
        plt.show()


if __name__ == "__main__":

    import sys
    import json
    import powerflow
    import powerflow.plotting as plt

    # Load original grid model:
    grid = powerflow.grids.eu_lv_feeder()
    S = powerflow.loads.eu_lv_feeder(grid)
    # plt.plotgraph(grid)

    # Lossless Reduction:
    grid, reduced_nodes, S = powerflow.grid_reduction.lossless.reduce(grid, S)
    # U_org, iters, runtime = powerflow.zbusjacobi(grid, S, verbose=True)
    # plt.plotgraph(grid)

    # Reduction to n nodes:
    n = 5
    print(f"\nReducing grid to {n} nodes")
    grid_red, S_red, nodes_reduced = reduce_to_n_nodes(grid, S, n)
    # U_red, iters_red, runtime_red = powerflow.zbusjacobi(grid_red, S_red)
    # U_red, iters_red, runtime_red = powerflow.ybusnewton(grid_red, S_red)
    # U_red, iters_red, runtime_red = powerflow.ybusgaussseidel(grid_red, S_red)
    # U_red, iters_red, runtime_red = powerflow.ybusjacobi(grid_red, S_red)
    plt.plotgraph(grid_red, S=S_red, filename=f"european_lv_feeder_red_to_{n}_2")
    # plot_voltage_comparison(U_red, U_org, f'european_lv_feeder_voltages_red_to_{n}')

    # Reduction to single feeder:
    # print(f'\nReducing grid to single feeder')
    # grid_feeder, S_feeder, nodes_reduced = reduce_to_single_feeder(grid, S)
    # U_feeder, iters_feeder, runtime_feeder = powerflow.ybusnewton(grid_feeder, S_feeder)
    # U_feeder, iters_feeder, runtime_feeder = powerflow.zbusjacobi(grid_feeder, S_feeder)
    # U_feeder, iters_feeder, runtime_feeder = powerflow.ybusgaussseidel(grid_feeder, S_feeder)
    # U_feeder, iters_feeder, runtime_feeder = powerflow.ybusjacobi(grid_feeder, S_feeder)
    # U_feeder, iters_feeder, runtime_feeder = powerflow.bfs(grid_feeder, S_feeder)
    # plt.plotgraph(grid_feeder,S=S_feeder)
    # plot_voltage_comparison(U_feeder, U_org)
    # plt.plotgraph(grid_feeder,S=S_feeder, filename='european_lv_feeder_singlefeeder')
    # plot_voltage_comparison(U_feeder, U_org, 'european_lv_feeder_voltages_singlefeeder')

    # Plot grid with path of highest impedance highlighted:
    # impedance_of_paths, path_to_nodes = find_impedance_paths(grid)
    # largest_impedance_path = sorted(impedance_of_paths.items(), key=lambda i:abs(i[1]))[-1]
    # final_node_name, _ = largest_impedance_path
    # path_of_largest_impedance = path_to_nodes[final_node_name]
    # node_id_to_node_name = {n['name']:n['id'] for n in grid['nodes']}
    # nodes_in_path = [node_id_to_node_name[node_name] for node_name in path_of_largest_impedance]

    # plotgraph_highlight_path(grid, S=S, nodes_in_path=nodes_in_path, filename='european_lv_feeder_highlightpath')
