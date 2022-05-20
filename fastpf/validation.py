import sys
import collections
import numpy as np
from .log import logger


def validate_grid(grid, S=None, logging_enabled=True, raise_on_error=False):
    """Check the grid dict for internal consistency"""

    logger.debug("Checking grid for validity...")
    error_messages = []  # we will now see about that

    def _add_to_errors(message):
        if logging_enabled:
            logger.error(message)
        error_messages.append(message)
        if raise_on_error:
            raise ValueError(message)

    # Check if the grid has a 'nodes'-list:
    if not "nodes" in grid:
        _add_to_errors('No "nodes" in grid')
    nodes = grid["nodes"]

    # Check if the grid has a 'nodes'-list:
    if not "lines" in grid:
        _add_to_errors('No "lines" in grid')
    lines = grid["lines"]

    # Check if the nodes-entries have at least an id:
    for key in ["id"]:
        for node in grid["nodes"]:
            if key not in node:
                _add_to_errors(f'{key} missing in grid["nodes"] for node {node}')

    # Check if the lines-entries have at least the necessary keys:
    for line in lines:
        for key in ["id", "from", "to", "R", "X"]:
            if key not in line:
                _add_to_errors(f'"{key}" missing in grid["lines"] for line {line}')

    # Check if node indices are properly sorted:
    number_of_nodes = len(nodes)
    node_index_list = []
    for node in nodes:
        node_index_list.append(node["id"])
    for n in range(number_of_nodes):
        if n not in node_index_list:
            _add_to_errors(f"Node ID {n} not found")
    for i in node_index_list:
        if n not in range(number_of_nodes):
            _add_to_errors(f"Node ID {n} not found")

    # Check if there are slack nodes in the grid:
    slacknodes = [n for n in nodes if n["is_slack"]]
    if not slacknodes:
        _add_to_errors("No slack node found in grid")

    # Check if slack nodes have a slack_voltage and slack_angle:
    for sn in slacknodes:
        if "slack_voltage" not in sn:
            _add_to_errors(
                'is_slack is True for Node {sn["id"]}, but no slack voltage is given'
            )
        if "slack_angle" not in sn:
            _add_to_errors(
                f'is_slack is True for Node {sn["id"]}, but no slack angle is given'
            )

    # Check if any of the lines lead to a node that does not exist:
    node_ids = [n["id"] for n in nodes]
    for line in lines:
        if line["from"] not in node_ids:
            _add_to_errors(
                f'Line {line["id"]} has source {line["from"]}, which is not a valid Node ID}}'
            )
        if line["to"] not in node_ids:
            _add_to_errors(
                f'Line {line["id"]} has target {line["to"]}, which is not a valid Node ID}}'
            )

    # Check if every node has a line attached to it:
    for n in nodes:
        connected = False
        for line in lines:
            if line["from"] == n["id"]:
                connected = True
                break
            if line["to"] == n["id"]:
                connected = True
                break
        if not connected:
            _add_to_errors(f"Node {n['id']} is not connected")

    # Check if S works with grid:
    if S is not None:
        if type(S) is not np.ndarray:
            _add_to_errors("type(S) is not numpy.ndarray")

        numberofloads, numberofnodes = S.shape
        if numberofnodes != len(nodes):
            _add_to_errors(
                f"Number of nodes between grid(n={len(nodes)}) and S(n={numberofnodes}) inconsistent"
            )

    if not error_messages:
        logger.debug("Validation: All checks passed")

    return error_messages


def is_radial(grid):

    # Shortcut:
    if len(grid["nodes"]) != len(grid["lines"]) + 1:
        return False

    sourcenodes = [min(l["source"], l["target"]) for l in grid["lines"]]
    sourcenodes_count = collections.Counter(sourcenodes).values()
    if not all([c == 1 for c in sourcenodes_count]):
        return True
    else:
        return False


def determine_gridshape(grid):
    """Determine the topology of the grid - feeder, radial, or meshed"""

    if len(grid["nodes"]) != len(grid["lines"]) + 1:
        is_radial = False
        is_feeder = False
        return is_feeder, is_radial

    # Check if Feeder: If a source node appears more than once, there is a junction
    sourcenodes = [min(l["source"], l["target"]) for l in grid["lines"]]
    sourcenodes_count = collections.Counter(sourcenodes).values()
    if not all([c == 1 for c in sourcenodes_count]):
        is_feeder = False
        is_radial = True
        return is_feeder, is_radial

    is_feeder = True
    is_radial = True

    return is_feeder, is_radial
