from math import ceil
import numpy as np

import shapely.affinity as affinity
from shapely.geometry import Polygon


class Entity:
    last_id = 1

    # Constructor
    def __init__(self, name, polygon, dd, full_geometry_acquired, uid=0):
        if uid == 0:
            self.uid = Entity.last_id
            Entity.last_id = Entity.last_id + 1
        else:
            self.uid = uid
        self.name = name
        self.polygon = polygon
        self.pose = [list(polygon.centroid.coords)[0][0],
                     list(polygon.centroid.coords)[0][1],
                     0.0]
        self.full_geometry_acquired = full_geometry_acquired

        self.inflated_polygon = None
        self._is_inflated_polygon_valid = False

        self.discrete_polygon = None
        self._is_discrete_polygon_valid = False

    def get_inflated_polygon(self, dd):
        if not self._is_inflated_polygon_valid:
            self._make_inflated_polygon(dd)
            self._is_inflated_polygon_valid = True
        return self.inflated_polygon

    def get_discrete_polygon(self, dd):
        if not self._is_discrete_polygon_valid:
            self._discretize(dd)
            self._is_discrete_polygon_valid = True
        return self.discrete_polygon

    def within(self, other_entity):
        return self.polygon.within(other_entity.polygon)

    def rotate(self, angle):
        # May be improved for cases with modulo 90-degrees rotations with specific update of discrete_polygon.
        self.polygon = affinity.rotate(self.polygon, angle, 'centroid')
        self.pose[2] = (self.pose[2] + angle) % 360
        self._is_inflated_polygon_valid = False
        self._is_discrete_polygon_valid = False

    def translate(self, xoff, yoff, dd):
        self.polygon = affinity.translate(self.polygon, xoff, yoff)
        self.pose[0], self.pose[1] = list(self.polygon.centroid.coords)[0][0], list(self.polygon.centroid.coords)[0][1]
        self._is_inflated_polygon_valid = False
        if (xoff / dd.res != 0.0) or (yoff / dd.res != 0.0):
            self._is_discrete_polygon_valid = False

    def _make_inflated_polygon(self, dd):
        if dd.inflation_radius == 0.0:
            self.inflated_polygon = self.polygon
        else:
            self.inflated_polygon = self.polygon.buffer(dd.inflation_radius)

    def _discretize(self, dd):
        min_x, min_y, max_x, max_y = self.inflated_polygon.bounds

        width, height = max_x - min_x, max_y - min_y

        d_width, d_height = int(ceil(width / dd.res)), int(ceil(height / dd.res))

        grid = np.zeros((d_width, d_height))

        for i in range(d_width):
            for j in range(d_height):
                cell = Polygon([(min_x + float(i) * dd.res, min_y + float(j) * dd.res),
                                (min_x + float(i+1) * dd.res, min_y + float(j) * dd.res),
                                (min_x + float(i+1) * dd.res, min_y + float(j+1) * dd.res),
                                (min_x + float(i) * dd.res, min_y + float(j+1) * dd.res)])
                if cell.intersects(self.polygon):
                    grid[i][j] = dd.cost_lethal
                elif cell.intersects(self.inflated_polygon):
                    grid[i][j] = dd.cost_inscribed

        self.discrete_polygon = grid
