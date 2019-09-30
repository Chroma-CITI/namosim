from math import ceil
import numpy as np
import utils

import shapely.affinity as affinity
from shapely.geometry import Polygon


class Thing:
    last_id = 1

    # Constructor
    def __init__(self, name, polygon, pose, full_geometry_acquired, uid=0):
        if uid == 0:
            self.uid = Thing.last_id
            Thing.last_id = Thing.last_id + 1
        else:
            self.uid = uid
        self.name = name
        self.polygon = polygon
        self.pose = pose
        self.full_geometry_acquired = full_geometry_acquired

        self.inflated_polygon = None
        self._is_inflated_polygon_valid = False

        self.discrete_polygon = None
        self._is_discrete_polygon_valid = False

        self.discrete_inflated_polygon = None
        self._is_discrete_inflated_polygon_valid = False

        self.discrete_cells_set = None
        self._is_discrete_cell_set_valid = False

        self.discrete_inflated_cells_set = None
        self._is_discrete_inflated_cell_set_valid = False

    def within(self, other_entity):
        return self.polygon.within(other_entity.polygon)

    def get_inflated_polygon(self, dd):
        if not self._is_inflated_polygon_valid:
            self._inflate_polygon(dd)
        return self.inflated_polygon

    def get_discrete_polygon(self, dd):
        if not self._is_discrete_polygon_valid:
            self._discretize_to_grid(dd)
        return self.discrete_polygon

    def get_discrete_inflated_polygon(self, dd):
        if not self._is_discrete_inflated_polygon_valid:
            self._discretize_to_grid(dd)
        return self.discrete_inflated_polygon

    def get_discrete_cells_set(self, dd):
        if not self._is_discrete_cell_set_valid:
            self._discretize_to_cell_sets(dd)
        return self.discrete_cells_set

    def get_discrete_inflated_cells_set(self, dd):
        if not self._is_discrete_inflated_cell_set_valid:
            self._discretize_to_cell_sets(dd)
        return self.discrete_inflated_cells_set

    def set_polygon(self, polygon, dd):
        self.polygon = polygon
        self.pose = [list(self.polygon.centroid.coords)[0][0],
                     list(self.polygon.centroid.coords)[0][1],
                     self.pose[2]]

        self._is_inflated_polygon_valid = False
        self._is_discrete_polygon_valid = False
        self._is_discrete_inflated_polygon_valid = False
        self._is_discrete_cell_set_valid = False
        self._is_discrete_inflated_cell_set_valid = False

    def rotate(self, angle):
        # May be improved for cases with modulo 90-degrees rotations with specific update of discrete_polygon.
        self.polygon = affinity.rotate(self.polygon, angle, 'centroid')
        self.pose[2] = (self.pose[2] + angle) % 360

        self._is_inflated_polygon_valid = False
        self._is_discrete_polygon_valid = False
        self._is_discrete_inflated_polygon_valid = False
        self._is_discrete_cell_set_valid = False
        self._is_discrete_inflated_cell_set_valid = False

    def translate(self, xoff, yoff, dd):
        # May be improved for cases where the translation is equal to a multiple of the resolution
        self.polygon = affinity.translate(self.polygon, xoff, yoff)
        self.pose[0], self.pose[1] = list(self.polygon.centroid.coords)[0][0], list(self.polygon.centroid.coords)[0][1]

        self._is_inflated_polygon_valid = False
        if (xoff / dd.res != 0.0) or (yoff / dd.res != 0.0):
            self._is_discrete_polygon_valid = False
            self._is_discrete_inflated_polygon_valid = False
        self._is_discrete_cell_set_valid = False
        self._is_discrete_inflated_cell_set_valid = False

    def _inflate_polygon(self, dd):
        if dd.inflation_radius == 0.0:
            self.inflated_polygon = self.polygon
        else:
            self.inflated_polygon = self.polygon.buffer(dd.inflation_radius)
        self._is_inflated_polygon_valid = True

    def _discretize_to_grid(self, dd):
        inflated_polygon = self.get_inflated_polygon(dd)
        min_x, min_y, max_x, max_y = inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y

        d_width, d_height = int(ceil(width / dd.res)), int(ceil(height / dd.res))

        discrete_polygon_grid = np.zeros((d_width, d_height))
        discrete_inflated_polygon_grid = np.zeros((d_width, d_height))

        for i in range(d_width):
            for j in range(d_height):
                cell = Polygon([(min_x + float(i) * dd.res, min_y + float(j) * dd.res),
                                (min_x + float(i+1) * dd.res, min_y + float(j) * dd.res),
                                (min_x + float(i+1) * dd.res, min_y + float(j+1) * dd.res),
                                (min_x + float(i) * dd.res, min_y + float(j+1) * dd.res)])
                if cell.intersects(self.polygon):
                    discrete_polygon_grid[i][j] = dd.cost_lethal
                    discrete_inflated_polygon_grid[i][j] = dd.cost_lethal
                elif cell.intersects(inflated_polygon):
                    discrete_inflated_polygon_grid[i][j] = dd.cost_inscribed

        self.discrete_polygon = discrete_polygon_grid
        self.discrete_inflated_polygon = discrete_inflated_polygon_grid
        self._is_discrete_polygon_valid = True
        self._is_discrete_inflated_polygon_valid = True

    def _discretize_to_cell_sets(self, dd):
        inflated_polygon = self.get_inflated_polygon(dd)
        min_x, min_y, max_x, max_y = inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y

        d_width, d_height = int(ceil(width / dd.res)), int(ceil(height / dd.res))

        discrete_cells_set = set()
        discrete_inflated_cells_set = set()

        min_cell_x = int(round((min_x - dd.grid_pose[0]) / dd.res))
        min_cell_y = int(round((min_y - dd.grid_pose[1]) / dd.res))
        max_cell_x = min_cell_x + d_width
        max_cell_y = min_cell_y + d_height

        i = 0
        for x in range(min_cell_x, max_cell_x):
            j = 0
            for y in range(min_cell_y, max_cell_y):
                cell_polygon = Polygon([(min_x + float(i) * dd.res, min_y + float(j) * dd.res),
                                        (min_x + float(i+1) * dd.res, min_y + float(j) * dd.res),
                                        (min_x + float(i+1) * dd.res, min_y + float(j+1) * dd.res),
                                        (min_x + float(i) * dd.res, min_y + float(j+1) * dd.res)])
                is_in_matrix = utils.is_in_matrix((x, y), dd.d_width, dd.d_height)
                if is_in_matrix:
                    if cell_polygon.intersects(self.polygon):
                        discrete_cells_set.add((x, y))
                        discrete_inflated_cells_set.add((x, y))
                    elif cell_polygon.intersects(inflated_polygon):
                        discrete_inflated_cells_set.add((x, y))
                j += 1
            i += 1

        self.discrete_cells_set = discrete_cells_set
        self.discrete_inflated_cells_set = discrete_inflated_cells_set
        self._is_discrete_cell_set_valid = True
        self._is_discrete_inflated_cell_set_valid = True
