from stinger import stinger_net
import numpy as np
from src.utils import utils


def two_d_to_one_d(coords, width):
    return coords[0] * width + coords[1]


def one_d_to_two_d(index, width):
    return index % width, index / width


def occupancy_grid_to_stream_and_monitor(occupancy_grid, neighborhood=utils.TAXI_NEIGHBORHOOD, threshold_value=1):
    stinger_stream = stinger_net.StingerStream(host='localhost', port=10102, strings=True, undirected=True)
    stinger_monitor = stinger_net.StingerMon("monitor")
    width, height = len(occupancy_grid), len(occupancy_grid[0])

    free_cells = zip(*np.where(occupancy_grid < threshold_value))

    # for cell in free_cells:
    #     stinger_stream.add_vertex_update(two_d_to_one_d(cell, width), 1)

    edges = set()
    for cell in free_cells:
        for i, j in neighborhood:
            neighbor = cell[0] + i, cell[1] + j
            if (utils.is_in_matrix(neighbor, width, height)
                    and occupancy_grid[neighbor[0]][neighbor[1]] < threshold_value
                    and (neighbor, cell) not in edges):
                edges.add((cell, neighbor))

    for edge in edges:
        stinger_stream.add_insert(
            str(two_d_to_one_d(edge[0], width)), str(two_d_to_one_d(edge[1], width)), insert_strings=True)

    stinger_stream.send_batch()

    stinger = stinger_monitor.stinger()

    # stinger.save_to_file("stinger.txt")  # Function is buggy...

    return stinger_stream, stinger_monitor


def update_graph(stinger_stream, occupancy_grid, freed_cells_set, invaded_cells_set,
                 neighborhood=utils.TAXI_NEIGHBORHOOD, threshold_value=1):
    width, height = len(occupancy_grid), len(occupancy_grid[0])

    edges_to_delete = set()
    for cell in invaded_cells_set:
        for i, j in neighborhood:
            neighbor = cell[0] + i, cell[1] + j
            if (utils.is_in_matrix(neighbor, width, height)
                    and occupancy_grid[neighbor[0]][neighbor[1]] < threshold_value
                    and (neighbor, cell) not in edges_to_delete):
                edges_to_delete.add((cell, neighbor))

    for edge in edges_to_delete:
        stinger_stream.add_delete(two_d_to_one_d(edge[0], width), two_d_to_one_d(edge[1], width))

    edges_to_add = set()
    for cell in freed_cells_set:
        for i, j in neighborhood:
            neighbor = cell[0] + i, cell[1] + j
            if (utils.is_in_matrix(neighbor, width, height)
                    and occupancy_grid[neighbor[0]][neighbor[1]] < threshold_value
                    and (neighbor, cell) not in edges_to_add):
                edges_to_add.add((cell, neighbor))

    for edge in edges_to_add:
        stinger_stream.add_insert(two_d_to_one_d(edge[0], width), two_d_to_one_d(edge[1], width))

    stinger_stream.send_batch()


# def get_graph_connected_components(stinger_monitor):
#
#     return [cc for cc in sorted(networkx.connected_components(graph))]
#
#
# def connected_components_to_grid(connected_components, occupancy_grid):
#     connected_components_grid = np.zeros(occupancy_grid.shape, dtype=np.int16)
#     component_id_counter = 1
#     for connected_component in connected_components:
#         for cell in connected_component:
#             connected_components_grid[cell[0]][cell[1]] = component_id_counter
#         component_id_counter += 1
#     return connected_components_grid


def main():
    test_array = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])
    stinger_stream, stinger_monitor = occupancy_grid_to_stream_and_monitor(test_array)
    print()
    # connected_components = get_graph_connected_components(graph)
    # connected_components_grid = connected_components_to_grid(connected_components, test_array)
    # print(connected_components)
    # print(connected_components_grid)


if __name__ == "__main__":
    main()