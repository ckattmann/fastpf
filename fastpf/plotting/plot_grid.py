import copy
import matplotlib.pyplot as plt
import networkx as nx

from ..log import logger


def plot_grid(
    grid,
    filename="",
    S=None,
    I=None,
    ax=None,
    shape=None,
    node_labels=False,
    lines_removed=[],
    verbose=False,
    **additional_args,
):

    grid = copy.deepcopy(grid)
    nodes = grid["nodes"]
    node_ids = [g["id"] for g in nodes]
    if verbose:
        print(f"Plotting {len(node_ids)} Nodes...")

    g = nx.OrderedGraph()
    g.add_nodes_from(node_ids)
    edge_color = []
    edgelist = []
    for line in sorted(grid["lines"], key=lambda line: line["id"]):
        g.add_edge(line["from"], line["to"])
        # Set Line Colors:
        edgelist.append(
            (line["from"], line["to"])
        )  # Explicit edgelist is required so that edge_color works properly
        if I is not None:
            Imax = np.max(np.abs(I))
            cmap = matplotlib.cm.get_cmap("OrRd")
            edge_color.append(cmap(np.max(np.abs(I[:, line["id"]])) / Imax))
            if line["id"] in lines_removed:
                edge_color[-1] = "blue"
        else:
            edge_color = "gray"

    # Set 'noload' for Node Colors and sizes:
    # =======================================
    if S is not None:
        for node in nodes:
            if np.sum(np.abs(S[:, node["id"]])) != 0:
                node["noload"] = False
            else:
                node["noload"] = True
    else:  # If S is not provided, noload=False for every node:
        for i, n in enumerate(nodes):
            nodes[i]["noload"] = False

    # Set Node Positions:
    # ===================
    if shape == "spring":
        positions = nx.spring_layout(g)
    elif shape == "force":
        starting_positions = None
        # if 'x' in nodes[0] and 'y' in nodes[0]:
        #     starting_positions = {n['id']:(n['x'], n['y']) for n in nodes}
        positions = nx.kamada_kawai_layout(g, pos=starting_positions)
    elif shape is None:
        if "x" in nodes[0] and "y" in nodes[0]:
            positions = {n["id"]: (n["x"], n["y"]) for n in nodes}
        elif "xy" in nodes[0]:
            positions = {n["id"]: (n["xy"][0], n["xy"][1]) for n in nodes}
        else:
            positions = nx.kamada_kawai_layout(g)

    # Set Slacks to green, loads to orange, rest to gray:
    # Filter nodes with PV and color blue:
    for i, node in enumerate(nodes):
        if "elements" in node:
            connected_element_types = [el["type"] for el in node["elements"]]
        else:
            connected_element_types = []
        if "PV" in connected_element_types:
            nodes[i]["has_pv"] = True
        else:
            nodes[i]["has_pv"] = False

    if "color" in nodes[0]:
        node_color = [n["color"] for n in nodes]
    else:
        node_color = [
            "orangered" if not n["noload"] else "dimgray" for n in grid["nodes"]
        ]
        node_color = [
            "blue" if n["has_pv"] else nc for n, nc in zip(grid["nodes"], node_color)
        ]
        node_color = [
            "green" if n["is_slack"] else nc for n, nc in zip(grid["nodes"], node_color)
        ]

    if "size" in nodes[0]:
        node_size = [n["size"] for n in nodes]
    else:
        node_size = [18 if not n["noload"] else 3 for n in grid["nodes"]]
        node_size = [
            18 if n["is_slack"] else ns for n, ns in zip(grid["nodes"], node_size)
        ]

    if ax == None:
        fig, ax = plt.subplots()
        # plt.setsize(fig, 1)
        ax.axis("off")
        plt.margins(0.01)
        ax.margins(0.02)
        showplot = True
    else:
        fig = None
        ax.axis("off")
        showplot = False

    nx.draw_networkx(
        g,
        ax=ax,
        pos=positions,
        node_color=node_color,
        node_size=node_size,
        edgelist=edgelist,
        edge_color=edge_color,
        width=1.5,
        with_labels=node_labels,
        **additional_args,
    )

    plt.tight_layout()

    if filename:
        if not fig:
            print("Cant save plot in external axes")
        else:
            save(fig, f"{filename}.eps")

    if showplot:
        plt.show()
