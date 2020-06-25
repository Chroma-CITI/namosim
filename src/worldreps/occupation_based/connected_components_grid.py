import networkx
import numpy as np
from src.utils import utils


class ConnectedComponentsMeta:
    def __init__(self, occupancy_grid, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD):
        self._occupancy_grid = occupancy_grid
        self._graph = occupancy_grid_to_graph(occupancy_grid, neighborhood)
        self._components = get_graph_connected_components(self._graph)
        self._grid = connected_components_to_grid(self._components, occupancy_grid)
        self._last_cc_id = max(self._components.keys())

        self._freed_cells = set()
        self._invaded_cells = set()
        self._neighborhood = neighborhood

        self._are_components_valid = True
        self._is_grid_valid = True

    def update_cells(self, freed_cells, invaded_cells):
        self._freed_cells.update(freed_cells)
        self._freed_cells.difference_update(invaded_cells)
        self._invaded_cells.update(invaded_cells)
        self._invaded_cells.difference_update(freed_cells)

        self._are_components_valid = False
        self._is_grid_valid = False

    def get_graph(self):
        if self._freed_cells or self._invaded_cells:
            update_graph(self._graph, self._occupancy_grid, self._freed_cells, self._invaded_cells, self._neighborhood)
        return self._graph

    def get_components(self):
        if not self._are_components_valid:
            self._components = get_graph_connected_components(self.get_graph(), self._components, self._last_cc_id)
            self._last_cc_id = max(self._last_cc_id, max(self._components.keys()))
            self._are_components_valid = True
        return self._components

    def get_grid(self):
        if not self._is_grid_valid:
            self._grid = connected_components_to_grid(self.get_components(), self._occupancy_grid)
            self._is_grid_valid = True
        return self._grid


def occupancy_grid_to_graph(occupancy_grid, neighborhood=utils.TAXI_NEIGHBORHOOD, threshold_value=1):
    graph = networkx.Graph()
    if isinstance(occupancy_grid, list):
        width, height = len(occupancy_grid), len(occupancy_grid[0])
    elif isinstance(occupancy_grid, np.ndarray):
        width, height = occupancy_grid.shape
    else:
        raise TypeError("occupancy_grid_to_graph method expects occupancy_grid of type list or numpy.ndarray")

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


def get_graph_connected_components(graph, prev_ccs=None, last_cc_id=1, priorized_component_id=-1):
    new_ccs = {counter: cc for counter, cc in enumerate(networkx.connected_components(graph), 1)}
    if not prev_ccs:
        return new_ccs
    else:
        final_ccs = dict()
        correspondence_numbers = dict()  # dict({-1: -1})
        identified_prev_ccs_ids = set()
        identified_new_ccs_ids = set()
        for new_id, new_cc in new_ccs.items():
            prev_ccs_keys_to_check = set(prev_ccs.keys()).difference(identified_prev_ccs_ids)
            for prev_id in prev_ccs_keys_to_check:
                prev_cc = prev_ccs[prev_id]
                if prev_cc == new_cc:
                    final_ccs[prev_id] = prev_cc
                    identified_prev_ccs_ids.add(prev_id)
                    identified_new_ccs_ids.add(new_id)
                    print("Component {} is unchanged".format(prev_id))
                    break
                else:
                    prev_new_intersection = new_cc.intersection(prev_cc)
                    if new_id not in correspondence_numbers:
                        correspondence_numbers[new_id] = {prev_id: len(prev_new_intersection)}
                    else:
                        correspondence_numbers[new_id][prev_id] = len(prev_new_intersection)
        for new_id, prev_id_to_nb in correspondence_numbers.items():
            if priorized_component_id in prev_id_to_nb and priorized_component_id not in identified_prev_ccs_ids:
              final_ccs[priorized_component_id] = new_ccs[new_id]
              identified_prev_ccs_ids.add(priorized_component_id)
              identified_new_ccs_ids.add(new_id)
              other_prev_ccs = set(prev_id_to_nb.keys()).difference({priorized_component_id})
              print("Component {} has fused with components {}.".format(priorized_component_id, str(other_prev_ccs)))

            else:
                closest_prev_id = max(prev_id_to_nb.iterkeys(), key=(lambda key: prev_id_to_nb[key]))
                if prev_id_to_nb[closest_prev_id] > 0 and closest_prev_id not in identified_prev_ccs_ids:
                    final_ccs[closest_prev_id] = new_ccs[new_id]
                    identified_prev_ccs_ids.add(closest_prev_id)
                    identified_new_ccs_ids.add(new_id)
                    other_prev_ccs = set(prev_id_to_nb.keys()).difference({closest_prev_id})
                    print("Component {} has fused with components {}.".format(closest_prev_id, str(other_prev_ccs)))

        deleted_ccs = set(prev_ccs.keys()).difference(identified_prev_ccs_ids)
        print("Components {} have been deleted.".format(str(deleted_ccs)))

        # TODO: Make it so scinded prev components are clearly named
        really_new_ccs_ids = set(new_ccs.keys()).difference(identified_new_ccs_ids)
        for really_new_cc_id in really_new_ccs_ids:
            last_cc_id += 1
            final_ccs[last_cc_id] = new_ccs[really_new_cc_id]
        return final_ccs


def connected_components_to_grid(connected_components, occupancy_grid):
    connected_components_grid = np.zeros(occupancy_grid.shape, dtype=np.int16)
    for component_id, connected_component in connected_components.items():
        for cell in connected_component:
            connected_components_grid[cell[0]][cell[1]] = component_id
    return connected_components_grid


if __name__ == "__main__":
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

    test_array = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])
    update_graph(graph, test_array, {(2,2)}, set())
    connected_components = get_graph_connected_components(graph, connected_components, 2, 2)
    connected_components_grid = connected_components_to_grid(connected_components, test_array)

    print(connected_components)
    print(connected_components_grid)

    test_array = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])
    update_graph(graph, test_array, set(), {(2,2)})
    connected_components = get_graph_connected_components(graph, connected_components, 2, 2)
    connected_components_grid = connected_components_to_grid(connected_components, test_array)

    print(connected_components)
    print(connected_components_grid)
