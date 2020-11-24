import unittest
import os
import progressbar
import snamosim.behaviors.algorithms.graph_search as graph_search
import numpy as np
from snamosim.utils import utils
import multiprocessing

import time
from pathfinding.core.grid import Grid

from snamosim.behaviors.algorithms import a_star_restart
NB_USABLE_CORES = multiprocessing.cpu_count() - 1

ASTAR_PBS = {
    ('hrt201d.map', 62),
    ('hrt201d.map', 43),
    ('hrt201d.map', 48),
    ('den012d.map', 47),
    ('hrt201d.map', 69),
    ('den012d.map', 50),
    ('den407d.map', 46),
    ('den407d.map', 60),
    ('lak203d.map', 32),
    ('hrt201d.map', 82),
    ('den407d.map', 65),
    ('den407d.map', 62),
    ('hrt201d.map', 83),
    ('den407d.map', 69),
    ('den407d.map', 68),
    ('den407d.map', 74),
    ('den012d.map', 92),
    ('den407d.map', 80),
    ('den407d.map', 88),
    ('den407d.map', 93),
    ('den012d.map', 96),
    ('den012d.map', 97),
    ('lak203d.map', 81),
    ('den407d.map', 97),
    ('den012d.map', 99),
    ('den407d.map', 99),
    ('hrt201d.map', 100),
    ('den101d.map', 30),
    ('lak203d.map', 86),
    ('arena2.map', 67),
    ('lak203d.map', 87),
    ('lak203d.map', 89),
    ('den407d.map', 106),
    ('den101d.map', 51),
    ('lgt603d.map', 121),
    ('ost003d.map', 61),
    ('den012d.map', 110),
    ('ost003d.map', 68)
}


def get_all_maps_names_and_filepaths(dirpath):
    return {
        f: {
            'map_path': os.path.join(dp, f),
            'scenarios_path': os.path.join(dp, '../', f + '-scen', f + '.scen'),
            'matrix': read_map_file(os.path.join(dp, f)),
            'scenarios_data': read_scenario_file(os.path.join(dp, '../', f + '-scen', f + '.scen'))
        }
        for dp, dn, filenames in os.walk(dirpath)
            for f in filenames
        if (
            os.path.isfile(os.path.join(dp, f))
            and os.path.splitext(f)[1] == '.map'
        )
    }


def read_map_file(map_filepath):

    """
    G - passable terrain
    . - passable terrain
    @ - out of bounds
    T - trees (unpassable)
    O - out of bounds
    S - swamp (passable from regular terrain)
    W - water (traversable, but not passable from terrain)
    :param map_filepath:
    :type map_filepath:
    :return:
    :rtype:
    """
    with open(map_filepath) as f:
        lines = f.readlines()
        height_line, width_line, map_lines = lines[1], lines[2], lines[4:]
        width = [int(s) for s in width_line.split() if s.isdigit()][0]  # = line length
        height = [int(s) for s in height_line.split() if s.isdigit()][0]  # = grid lines count
        grid = np.zeros((width, height), dtype=int)
        for y, line in enumerate(map_lines):
            for x, letter in enumerate(line):
                if letter == '.':
                    pass
                elif letter == 'G':
                    pass
                elif letter == '@':
                    grid[x][y] = 1
                elif letter == 'O':
                    grid[x][y] = 1
                elif letter == 'T':
                    grid[x][y] = 1
                elif letter == 'S':
                    NotImplementedError(
                        'Swamp letter not yet implemented.'
                    )
                elif letter == 'W':
                    NotImplementedError(
                        'Water letter not yet implemented.'
                    )
                else:
                    ValueError(
                        'Only the letters [., G, @, O, T, S, W] are valid, found letter <{}>.'.format(letter)
                    )
        return grid


def read_scenario_file(scenario_filepath):
    with open(scenario_filepath) as f:
        lines = f.readlines()[1:]
        return [Scenario(*(line.split('\t'))) for line in lines if line]


class Scenario:
    def __init__(self, bucket, map_name, width, height, start_x, start_y, goal_x, goal_y, length):
        self.bucket = int(bucket)
        self.map_name = map_name
        self.width = int(width)
        self.height = int(height)
        self.start_cell = (int(start_x), int(start_y))
        self.goal_cell = (int(goal_x), int(goal_y))
        self.length = float(length)

    def __str__(self):
        return 'Start cell: {}, Goal cell: {}, Expected length: {}'.format(self.start_cell, self.goal_cell, self.length)


class GraphSearchTest(unittest.TestCase):
    def setUp(self):
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        self.benchmarks_dirname = os.path.join(
            self.dirname, '../../../data/gridsearch-dataset/www.movingai.com/benchmarks'
        )
        self.dao_dataset_path = os.path.join(self.benchmarks_dirname, 'dao')
        print('Loading dao dataset...')
        self.dao_dataset_maps = get_all_maps_names_and_filepaths(self.dao_dataset_path)
        print('...Loaded dao dataset.')

    def for_map(self, map_name, my_map, map_counter):

        # print('Testing for map: {}'.format(map_name))
        # scenario_progressbar = progressbar.ProgressBar(max_value=len(map['scenarios_data']))
        for scenario_counter, scenario in enumerate(my_map['scenarios_data']):
            self.for_scenario(map_name, my_map, scenario, scenario_counter)
        # print('Finished testing map: {}'.format(map_name))

        # map_progressbar.update(map_counter)
    def for_scenario(self, map_name, my_map, scenario, scenario_counter,
                     # path_finding_algorithm=None,
                     path_finding_algorithm=graph_search.grid_search_a_star,
                     log_start=False, log_success=False):
        if log_start:
            print('Testing for map {}, scenario {} : {}'.format(map_name, scenario_counter, str(scenario)))

        sends_back_all_data = (
            path_finding_algorithm == graph_search.grid_search_dijkstra
            or path_finding_algorithm == graph_search.grid_search_a_star
        )

        if sends_back_all_data:
            path_found, end_cell, came_from, _, gscore, _ = path_finding_algorithm(
                start=scenario.start_cell,
                goal=scenario.goal_cell,
                grid=my_map['matrix'],
                width=my_map['matrix'].shape[0],
                height=my_map['matrix'].shape[1],
                chess_neighborhood=True
            )
            if path_found:
                raw_path = graph_search.reconstruct_path(came_from, end_cell)
                measured_length = sum(
                    [utils.chebyshev_distance(cur_cell, raw_path[i + 1]) for i, cur_cell in enumerate(raw_path[:-1])]
                )
                try:
                    self.assertAlmostEqual(scenario.length, measured_length, places=2)
                    if log_success:
                        print('Map {}, Scenario {} successful.'.format(map_name, scenario_counter, str(scenario)))
                except AssertionError as e:
                    print(
                        'Path found for Map {}, Scenario {}, but found cost <{}> not equal to expected cost <{}>'.format(
                            map_name, scenario_counter, measured_length, scenario.length
                        )
                    )
            else:
                try:
                    self.assertTrue(path_found)
                except AssertionError as e:
                    print('Path could not be found for Map {}, Scenario {}.'.format(map_name, scenario_counter))
        else:
            path = a_star_restart.find_path(scenario.start_cell, scenario.goal_cell, my_map['matrix'])
            measured_length = sum(
                [utils.chebyshev_distance(cur_cell, path[i + 1]) for i, cur_cell in enumerate(path[:-1])]
            )
            try:
                self.assertAlmostEqual(scenario.length, measured_length, places=2)
                if log_success:
                    print('Map {}, Scenario {} successful.'.format(map_name, scenario_counter, str(scenario)))
            except AssertionError as e:
                print(
                    'Path found for Map {}, Scenario {}, but found cost <{}> not equal to expected cost <{}>'.format(
                        map_name, scenario_counter, measured_length, scenario.length
                    )
                )

        # scenario_progressbar.update(scenario_counter)

    def test_dao_dataset_for_a_star(self, multiprocess=True):
        # map_progressbar = progressbar.ProgressBar(max_value=len(self.dao_dataset_maps))
        print('Testing dao dataset...')
        processes = set()
        for map_counter, (map_name, my_map) in enumerate(self.dao_dataset_maps.items()):
            if multiprocess:
                while len(processes) >= NB_USABLE_CORES:
                    processes_to_remove = []
                    for p in processes:
                        if not p.is_alive():
                            processes_to_remove.append(p)
                    for p in processes_to_remove:
                        p.terminate()
                        processes.remove(p)
                    time.sleep(0.2)
                p = multiprocessing.Process(target=self.for_map, args=(map_name, my_map, map_counter))
                processes.add(p)
                p.start()
            else:
                self.for_map(map_name, my_map, map_counter)
        print('...Tested dao dataset.')


if __name__ == '__main__':

    unittest.main()
