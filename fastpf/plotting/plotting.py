import os
import copy
import networkx as nx

from matplotlib import rc
import matplotlib.font_manager

rc("text", usetex=True)
rc("font", family="serif")
# required for an upright, non-italic mu:
rc("text.latex", preamble=r"\usepackage{upgreek}")

# This allows import fastpf.plotting as plt:
from matplotlib.pyplot import *

# TODO: potentally not zip-safe, needs to be done with pkg_resources:
style.use(os.path.join(os.path.dirname(__file__), "diss.mplstyle"))

cmap = get_cmap("Blues")
cmap_greens = get_cmap("Greens")

# Bold font:
boldfont = {"fontname": "CMU Serif"}

# Gray box for legends and text boxes:
graybox = {"boxstyle": "round", "linewidth": 0, "facecolor": "lightgray"}

# Helper Function so you can set size as a multiplicator of 2.5 inches with plt.setsize(fig,1...n)
def setsize(fig, size=1):
    fig.set_size_inches((4.33, 2.5 * size))


# Helper Function to reach the images folder:
to_images = functools.partial(
    os.path.join, os.getenv("HOME"), "Dropbox", "diss", "images"
)


def save(fig, filename):
    if "." not in filename:
        filename += ".eps"
    fig.savefig(to_images(filename), dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Figure saved to {filename}")


def plotgraph(
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
    g = nx.OrderedGraph()
    nodes = grid["nodes"]
    node_ids = [g["id"] for g in nodes]
    if verbose:
        print(f"Plotting {len(node_ids)} Nodes...")

    g.add_nodes_from(node_ids)
    edge_color = []
    edgelist = []
    for line in sorted(grid["edges"], key=lambda line: line["id"]):
        g.add_edge(line["source"], line["target"])
        # Set Line Colors:
        edgelist.append(
            (line["source"], line["target"])
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
        fig, ax = subplots()
        setsize(fig, 1)
        ax.axis("off")
        margins(0.01)
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

    tight_layout()

    if filename:
        if not fig:
            print("Cant save plot in external axes")
        else:
            save(fig, f"{filename}.eps")

    if showplot:
        show()


def plot_result(grid, S, U, I, lines_removed=[], verbose=False):

    if verbose:
        for line in grid["edges"]:
            print(
                f'Line {line["id"]:>3}:   {line["source"]:>3} - {line["target"]:>3}:    R,X = {line["R"]:7.3f}, {line["X"]:7.3f}    I = {abs(I)[:,line["id"]].max():.2f}'
            )

    fig = figure()
    gs = matplotlib.gridspec.GridSpec(3, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("Abs. Power S / VA")
    plot(np.sign(S.real) * np.abs(S))
    ax0.grid(True)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title("Abs. Voltage / V")
    plot(np.abs(U), alpha=0.5)
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[2, 0])
    bar(range(I.shape[1]), np.max(np.abs(I), axis=0), zorder=2)
    # plt.plot(np.sign(I.real) * np.abs(I))
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[:, 1])
    plotgraph(grid, I=I, S=S, ax=ax3, node_labels=False, lines_removed=lines_removed)

    tight_layout()
    show()
