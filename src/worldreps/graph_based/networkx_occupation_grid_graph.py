import networkx
import numpy as np
from src.utils import utils


def occupancy_grid_to_graph(occupancy_grid, neighborhood=utils.TAXI_NEIGHBORHOOD, threshold_value=1):
    graph = networkx.Graph()
    width, height = len(occupancy_grid), len(occupancy_grid[0])

    free_cells = zip(*np.where(occupancy_grid < threshold_value))
    graph.add_nodes_from(free_cells)

    edges = set()
    for cell in free_cells:
        for i, j in neighborhood:
            neighbor = cell[0] + i, cell[1] + j
            if (utils.is_in_matrix(neighbor, width, height)
                    and occupancy_grid[neighbor[0]][neighbor[1]] < threshold_value
                    and (neighbor, cell) not in edges):
                edges.add((cell, neighbor))
    graph.add_edges_from(edges)
    return graph


def update_graph(graph, occupancy_grid, freed_cells_set, invaded_cells_set,
                 neighborhood=utils.TAXI_NEIGHBORHOOD, threshold_value=1):
    graph.remove_nodes_from(invaded_cells_set)

    graph.add_nodes_from(freed_cells_set)

    width, height = len(occupancy_grid), len(occupancy_grid[0])
    edges = set()
    for cell in freed_cells_set:
        for i, j in neighborhood:
            neighbor = cell[0] + i, cell[1] + j
            if (utils.is_in_matrix(neighbor, width, height)
                    and occupancy_grid[neighbor[0]][neighbor[1]] < threshold_value
                    and (neighbor, cell) not in edges):
                edges.add((cell, neighbor))
    graph.add_edges_from(edges)


def get_graph_connected_components(graph):
    return [cc for cc in sorted(networkx.connected_components(graph))]


def connected_components_to_grid(connected_components, occupancy_grid):
    connected_components_grid = np.zeros(occupancy_grid.shape, dtype=np.int16)
    component_id_counter = 1
    for connected_component in connected_components:
        for cell in connected_component:
            connected_components_grid[cell[0]][cell[1]] = component_id_counter
        component_id_counter += 1
    return connected_components_grid


def main():
    test_array = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])
    graph = occupancy_grid_to_graph(test_array)
    connected_components = get_graph_connected_components(graph)
    connected_components_grid = connected_components_to_grid(connected_components, test_array)
    print(connected_components)
    print(connected_components_grid)


if __name__ == "__main__":
    main()
