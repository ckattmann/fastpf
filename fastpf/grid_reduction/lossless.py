import json
import queue
import copy
import numpy as np
import networkx as nx

import pprint

def reduce(grid, S=None, verbose=False):
    ''' Remove electrically irrelevant nodes from a grid
    '''

    if verbose:
        log = print
    else:
        log = lambda *args, **kwargs: None

    # Avoid changing the grid in-place:
    grid = copy.deepcopy(grid)

    nodes = grid['nodes']
    lines = grid['edges']
    node_deleted_last_run = True
    nodes_deleted = []
    # Try deleting nodes until no unnecessary node left:
    while node_deleted_last_run:
        node_deleted_last_run = False
        for node_i, node in enumerate(nodes):
            node_id = node['id']

            # Shortcut continue for loads and slacks, obviously not removable:
            if 'is_slack' in node.keys() and node['is_slack']:
                log(f'Node {node["id"]} is slack, skipping')
                continue
            if S is None:
                if 'noload' in node.keys() and node['noload'] == False:
                    continue
            else:
                if np.any(np.abs(S[:,node['id']]) > 0):
                    log(f'Node {node["id"]} has load {np.max(np.abs(S)[:,node["id"]])}, skipping')
                    continue

            # Find connected lines and nodes:
            connected_nodes = []
            connected_lines = []
            line_indices = []
            for line_i, line in enumerate(lines):
                if node['id'] == line['source']:
                    connected_nodes.append(line['target'])
                    connected_lines.append(line)
                    line_indices.append(line_i)
                elif node['id'] == line['target']:
                    connected_nodes.append(line['source'])
                    connected_lines.append(line)
                    line_indices.append(line_i)

            # End Node -> Delete Node and Line:
            if len(connected_nodes) == 1:
                log(f'Deleting End Node {node["id"]}')
                nodes_deleted.append(node['id'])
                del nodes[node_i]
                del lines[line_indices[0]]
                node_deleted_last_run = True

            # Transit Node -> Add up connected lines and delete node:
            elif len(connected_nodes) == 2:
                assert len(connected_lines) == 2
                log(f'Deleting Transit Node {node["id"]}')
                nodes_deleted.append(node['id'])
                new_line = {}
                new_line['id'] = connected_lines[0]['id']
                new_line['source'] = connected_nodes[0]
                new_line['target'] = connected_nodes[1]
                if 'length_m' in connected_lines[0].keys() and 'length_m' in connected_lines[1].keys():
                    new_line['length_m'] = connected_lines[0]['length_m'] + connected_lines[1]['length_m']
                new_line['R'] = connected_lines[0]['R'] + connected_lines[1]['R']
                new_line['X'] = connected_lines[0]['X'] + connected_lines[1]['X']

                del nodes[node_i]
                for i in reversed(sorted(line_indices)):
                    del lines[i]
                lines.append(new_line)
                node_deleted_last_run = True

            else:
                log(f'Node {node["id"]} has {len(connected_nodes)} connected nodes, skipping')


    # Construct new grid dict:
    reduced_grid = {'nodes': nodes, 'edges': lines, 'grid': grid['grid']}

    # return reduced_grid, 0, S
    reduced_grid_normalised, S_normalized = normalise_node_ids(reduced_grid, S)

    # Delete reduced nodes from loads S:
    if S is not None:
        S_normalized_reduced = S_normalized[:,:len(nodes)]
    else:
        S_normalized_reduced = None
    return reduced_grid_normalised, nodes_deleted, S_normalized_reduced




def normalise_node_ids(grid, S=None):

    grid = copy.deepcopy(grid)

    nodes = grid['nodes']
    lines = grid['edges']

    node_id_dict = {n['id']:n for n in nodes}
    old_id_to_new_id = {}

    node_queue = queue.Queue()
    handled_nodes = []
    node_ids_in_queue = []
    node_id_counter = 0

    # Find first slack node to start:
    # print(nodes)
    for n in nodes:
        if n['is_slack']:
            node_queue.put(n)
            break

    while node_queue.qsize():
        node = node_queue.get()
        node_id = node['id']
        if node_id in handled_nodes:
            continue
        handled_nodes.append(node_id)
        for line in lines:
            if line['source'] == node_id:
                node_queue.put(node_id_dict[line['target']])
                node_ids_in_queue.append(line['target'])
            if line['target'] == node_id:
                node_queue.put(node_id_dict[line['source']])
                node_ids_in_queue.append(line['target'])
        old_id_to_new_id[node_id] = node_id_counter
        node_id_counter += 1

    if S is not None:
        Snew = np.zeros_like(S)
        for n in nodes:
            Snew[:,old_id_to_new_id[n['id']]] = S[:,n['id']]
    else:
        Snew = None

    for n in nodes:
        n['id'] = old_id_to_new_id[n['id']]

    for line_id, line in enumerate(lines):
        line['id'] = line_id
        line['source'] = old_id_to_new_id[line['source']]
        line['target'] = old_id_to_new_id[line['target']]

    grid = {'nodes': nodes, 'edges': lines, 'grid': grid['grid']}

    return grid, Snew






### DEV CODE ###

def reduce_intermediate_nodes(grid):
    nodes = grid['nodes']
    lines = grid['edges']
    node_deleted = True
    nodes_reduced = []
    while node_deleted:
        node_deleted = False
        for node_i, node in enumerate(nodes):  # reversed because generally, "chains" of reducable nodes start at the end
            if not node['noload'] or node['is_slack']:
                continue
            connected_nodes = []
            connected_lines = []
            line_indices = []
            for line_i, line in enumerate(lines):
                if node['id'] == line['source']:
                    connected_nodes.append(line['target'])
                    connected_lines.append(line)
                    line_indices.append(line_i)
                elif node['id'] == line['target']:
                    connected_nodes.append(line['source'])
                    connected_lines.append(line)
                    line_indices.append(line_i)

            if len(connected_nodes) == 2 and node['noload'] and not node['is_slack']:
                nodes_reduced.append(node['id'])
                new_line = {}
                new_line['id'] = connected_lines[0]['id']
                new_line['source'] = connected_nodes[0]
                new_line['target'] = connected_nodes[1]
                new_line['length_m'] = connected_lines[0]['length_m'] + connected_lines[1]['length_m']
                new_line['R'] = connected_lines[0]['R'] + connected_lines[1]['R']
                new_line['X'] = connected_lines[0]['X'] + connected_lines[1]['X']

                # Actually delete the nodes:
                del nodes[node_i]
                for i in reversed(sorted(line_indices)):
                    del lines[i]
                lines.append(new_line)
                node_deleted = True
                break
    grid = {'nodes': nodes, 'edges': lines, 'grid': grid['grid']}

    return grid, nodes_reduced



def reduce_end_nodes(grid):
    nodes = grid['nodes']
    lines = grid['edges']
    node_deleted = True
    nodes_reduced = []
    while node_deleted:
        node_deleted = False
        for node_i, node in enumerate(nodes):  # reversed because generally,
            connected_nodes = []
            line_indices = []
            for line_i, line in enumerate(lines):
                if node['id'] == line['source']:
                    connected_nodes.append(line['target'])
                    line_indices.append(line_i)
                elif node['id'] == line['target']:
                    connected_nodes.append(line['source'])
                    line_indices.append(line_i)
                if len(connected_nodes) > 1:
                    break

            if len(connected_nodes) == 1 and node['noload'] and not node['is_slack']:
                nodes_reduced.append(node['id'])
                del nodes[node_i]
                del lines[line_indices[0]]
                node_deleted = True
                break

    grid = {'nodes': nodes, 'edges': lines, 'grid': grid['grid']}
    return grid, nodes_reduced



def reduce_grid(grid):

    nodes_reduced_end = 1
    nodes_reduced_intermediate = 1

    while nodes_reduced_intermediate  or nodes_reduced_end:
        print('End Reduction...')
        grid, nodes_reduced_end = reduce_end_nodes(grid)
        print(f'Reduced {nodes_reduced_end} Nodes by end_reduction -> Nodes: {len(grid["nodes"])}, Lines: {len(grid["edges"])}')
        if nodes_reduced_end:
            plotgraph(grid)

        print('Intermediate Reduction...')
        grid, nodes_reduced_intermediate = reduce_intermediate_nodes(grid)
        print(f'Reduced {nodes_reduced_intermediate} Nodes by intermediate_reduction -> Nodes: {len(grid["nodes"])}, Lines: {len(grid["edges"])}')
        if nodes_reduced_intermediate:
            plotgraph(grid)

    return grid

if __name__ == '__main__':

    with open('european_lv_feeder.json','r') as f:
        grid = json.load(f)

    print(f'Nodes: {len(grid["nodes"])}, Lines: {len(grid["edges"])}')

    plotgraph(grid, filename='europeanLvFeeder_nonreduced')

    grid, nodes_reduced = reduce_grid(grid)

    plotgraph(grid, filename='europeanLvFeeder_reduced')

    with open('european_lv_feeder_reduced.json', 'w') as f:
        json.dump(grid, f)

