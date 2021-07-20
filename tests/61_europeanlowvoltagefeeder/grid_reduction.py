import json
import queue
import copy
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt

import powerflow.plotting as plt

def plotgraph(grid, filename='', **additional_args):
    g = nx.Graph()
    nodes = [g['id'] for g in grid['nodes']]
    positions = {n['id']:(n['x'], n['y']) for n in grid['nodes']}
    g.add_nodes_from(nodes)
    for e in grid['edges']:
        g.add_edge(e['source'],e['target'])

    # Set Slacks to green, loads to orange, rest to gray:
    node_color = ['orangered' if not n['noload'] else 'dimgray' for n in grid['nodes']]
    node_color = ['green' if n['is_slack'] else nc for n,nc in zip(grid['nodes'], node_color)]

    node_size = [18 if not n['noload'] else 3 for n in grid['nodes']]
    node_size = [18 if n['is_slack'] else ns for n,ns in zip(grid['nodes'], node_size)]

    fig, ax = plt.subplots(frameon=False)
    ax.axis('off')
    plt.setsize(fig, 1)

    nx.draw_networkx(g, pos=positions, node_color=node_color, node_size=node_size, edge_color='gray', width=1.5, with_labels=False, **additional_args)

    plt.margins(0.01)
    ax.margins(0.02)

    plt.tight_layout()

    if filename:
        plt.save(fig, f'{filename}.eps')
    plt.show()


def reduce_unnecessary_nodes(grid):
    ''' Remove electrically irrelevant nodes from a grid
    '''

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
            # Shortcut continue for loads and slacks, obviously not removable:
            if not node['noload'] or node['is_slack']:
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
                nodes_deleted.append(node['id'])
                del nodes[node_i]
                del lines[line_indices[0]]
                node_deleted_last_run = True
                break

            # Transit Node -> Add up connected lines and delete node:
            if len(connected_nodes) == 2:
                nodes_deleted.append(node['id'])
                new_line = {}
                new_line['id'] = connected_lines[0]['id']
                new_line['source'] = connected_nodes[0]
                new_line['target'] = connected_nodes[1]
                new_line['length_m'] = connected_lines[0]['length_m'] + connected_lines[1]['length_m']
                new_line['R'] = connected_lines[0]['R'] + connected_lines[1]['R']
                new_line['X'] = connected_lines[0]['X'] + connected_lines[1]['X']

                del nodes[node_i]
                for i in reversed(sorted(line_indices)):
                    del lines[i]
                lines.append(new_line)
                node_deleted_last_run = True
                break

    # Construct new grid dict:
    reduced_grid = {'nodes': nodes, 'edges': lines, 'grid': grid['grid']}

    return reduced_grid, nodes_deleted



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


def reduce_grid(grid, verbose=False, plot_graph=False):

    grid = copy.deepcopy(grid)

    nodes_reduced_end = []
    nodes_reduced_intermediate = []
    nodes_reduced = []

    first_run = True
    while first_run or nodes_reduced_intermediate or nodes_reduced_end:
        first_run = False

        grid, nodes_reduced_end = reduce_end_nodes(grid)
        nodes_reduced += nodes_reduced_end

        if verbose:
            print(f'Reduced {len(nodes_reduced_end)} Nodes by end_reduction -> Nodes: {len(grid["nodes"])}, Lines: {len(grid["edges"])}')
        if plot_graph and nodes_reduced_end:
            plotgraph(grid)

        grid, nodes_reduced_intermediate = reduce_intermediate_nodes(grid)
        nodes_reduced += nodes_reduced_intermediate

        if verbose:
            print(f'Reduced {len(nodes_reduced_intermediate)} Nodes by intermediate_reduction -> Nodes: {len(grid["nodes"])}, Lines: {len(grid["edges"])}')
        if plot_graph and nodes_reduced_intermediate:
            plotgraph(grid)

    return grid, nodes_reduced


def normalize_node_ids(grid):

    grid = copy.deepcopy(grid)

    nodes = grid['nodes']
    lines = grid['edges']
    
    node_id_dict = {n['id']:n for n in nodes}
    old_id_to_new_id = {}

    node_queue = queue.Queue()
    handled_nodes = []
    node_id_counter = 0

    # Find first slack node to start:
    for n in nodes:
        if n['is_slack']:
            node_queue.put(n)
            break

    while node_queue.qsize():
        node = node_queue.get()
        node_id = node['id']
        for line in lines:
            if line['source'] == node_id:
                if line['target'] not in handled_nodes:
                    node_queue.put(node_id_dict[line['target']])
            if line['target'] == node_id:
                if line['source'] not in handled_nodes:
                    node_queue.put(node_id_dict[line['source']])
        handled_nodes.append(node_id)
        old_id_to_new_id[node_id] = node_id_counter
        node_id_counter += 1

    for n in nodes:
        n['id'] = old_id_to_new_id[n['id']]
    for line in lines:
        line['source'] = old_id_to_new_id[line['source']]
        line['target'] = old_id_to_new_id[line['target']]

    grid = {'nodes': nodes, 'edges': lines, 'grid': grid['grid']}
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
