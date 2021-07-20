import json
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


def reduce_intermediate_nodes(grid):
    nodes = grid['nodes']
    lines = grid['edges']
    node_deleted = True
    nodes_reduced = 0
    while node_deleted:
        node_deleted = False
        for node_i, node in enumerate(nodes):
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
                # print(f'Reducing node {node["id"]}')
                nodes_reduced += 1
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
    nodes_reduced = 0
    while node_deleted:
        node_deleted = False
        for node_i, node in enumerate(nodes):
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

            if len(connected_nodes) == 1 and node['noload'] and not node['is_slack']:
                # print(f'Reducing node {node["id"]}')
                nodes_reduced += 1
                # Actually delete the nodes:
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

    grid = reduce_grid(grid)

    plotgraph(grid, filename='europeanLvFeeder_reduced')

    with open('newgrid.json', 'w') as f:
        json.dump(grid, f)
