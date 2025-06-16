import random
import typing as t
from xml.dom import minidom

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import Polygon
import shapely.ops

from namosim.data_models import NamoConfigModel
from namosim.mapgen import utils
from namosim.mapgen.connected_components import ConnectedComponents
from namosim.mapgen.types import FLOOR, PERM_WALL, WALL, GridCell
from namosim.utils.conversion import concave_hull_polygon
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from namosim.world.entity import Movability, Style
from namosim.world.obstacle import Obstacle
from namosim.world.world import World


class MapGen:
    def __init__(self, height: int, width: int, init_open: float = 0.40):
        self.height = height
        self.width = width
        self.area = height * width
        self.map: npt.NDArray[t.Any] = np.ones((height, width), dtype=np.int8) * WALL
        self.center: GridCell = (int(self.height / 2), int(self.width / 2))
        self.init(init_open)

    def print_grid(self):
        for r in self.map:
            row = ""
            for e in r:
                if e in (WALL, PERM_WALL):
                    row += "#"
                else:
                    row += "."
            print(row)

    def gen_map(self):
        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                wall_count = self.get_num_adj_walls(r, c)

                if self.map[r][c] == FLOOR:
                    if wall_count > 5:
                        self.map[r][c] = WALL
                elif self.map[r][c] != PERM_WALL and wall_count < 4:
                    self.map[r][c] = FLOOR

        self.components = ConnectedComponents(self.map)
        self.join_components()
        self._crack(start=(0, 0))
        self._crack(start=(self.height - 1, self.width - 1))
        return self.map

    def __set_border(self):
        for j in range(0, self.height):
            self.map[j][0] = PERM_WALL
            self.map[j][self.width - 1] = PERM_WALL

        for j in range(0, self.width):
            self.map[0][j] = PERM_WALL
            self.map[self.height - 1][j] = PERM_WALL

    def init(self, initial_open: float):
        open_count = int(self.area * initial_open)
        self.__set_border()

        walls: t.List[GridCell] = []
        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                walls.append((r, c))
        not_picked = set(range(0, len(walls)))

        for _ in range(open_count):
            [x] = random.sample(list(not_picked), 1)
            (r, c) = walls[x]
            not_picked.remove(x)
            self.map[r][c] = 0

    def get_num_adj_walls(self, sr: int, sc: int):
        count = 0

        for r in (-1, 0, 1):
            for c in (-1, 0, 1):
                if self.map[(sr + r)][sc + c] != FLOOR and not (r == 0 and c == 0):
                    count += 1

        return count

    def join_components(self):
        map = self.map
        cc = ConnectedComponents(map)
        while len(cc.components) > 1:
            for component in cc.components.keys():
                nn, path = cc.find_nearest_neighbor(component_idx=component)
                if nn is not None:
                    for cell in path:
                        map[cell[0]][cell[1]] = FLOOR
                    break
            cc = ConnectedComponents(map)

        self.map = map

    def _crack(self, start: GridCell):
        """Makes sur the map doesn't contain a 'hole' by opening a path from (0,0) to the floor cells."""

        def clear_path(end: GridCell, d: t.Dict[GridCell, GridCell]):
            self.map[end[0]][end[1]] = FLOOR
            while end in d:
                end = d[end]
                self.map[end[0]][end[1]] = FLOOR

        def get_neighbors(
            cell: GridCell, map: npt.NDArray[t.Any]
        ) -> t.Iterable[GridCell]:
            (r, c) = cell
            candidates = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            for cand in candidates:
                if not utils.is_in_map(cand[0], cand[1], map):
                    continue

                yield cand

        frontier: t.List[GridCell] = [start]
        visited: t.Set[GridCell] = set()
        came_from: t.Dict[GridCell, GridCell] = {}

        while len(frontier) > 0:
            x = frontier.pop(0)

            if self.map[x[0]][x[1]] == FLOOR:
                clear_path(x, came_from)
                return

            visited.add(x)
            for n in get_neighbors(x, self.map):
                if n in visited:
                    continue
                frontier.append(n)
                came_from[n] = x

    def to_svg(self, robot_radius_cm: float) -> minidom.Document:
        w = self.to_namo_world(robot_radius_cm=robot_radius_cm)
        return w.to_svg()

    def get_wall_polygons(self, cell_size: float) -> t.List[Polygon]:
        walls: t.List[Polygon] = []

        cc = ConnectedComponents(self.map, component_cell_types={WALL, PERM_WALL})

        for comp in cc.components.values():
            comp_polygons = []
            for r, c in comp:
                y_offset = r * cell_size
                x_offset = c * cell_size
                polygon = Polygon(
                    [
                        (x_offset, y_offset),
                        (x_offset, y_offset + cell_size),
                        (x_offset + cell_size, y_offset + cell_size),
                        (x_offset + cell_size, y_offset),
                        (x_offset, y_offset),
                    ]
                )

                comp_polygons.append(polygon)  # type: ignore
            full_poly = t.cast(Polygon, shapely.ops.unary_union(comp_polygons))  # type: ignore
            # full_poly = svg_utils.perturb_polygon(
            #     polygon=full_poly, bounds=(0, 0, width, height)
            # )

            if not full_poly.interiors:
                # pass
                full_poly = concave_hull_polygon(full_poly, alpha=0.03)
            else:
                smoothed_holes = []
                for hole in full_poly.interiors:
                    smoothed = Polygon(hole.coords).exterior.coords
                    smoothed = concave_hull_polygon(
                        Polygon(hole.coords), alpha=0.02
                    ).exterior.coords  # type: ignore
                    smoothed_holes.append(smoothed)
                full_poly = Polygon(full_poly.exterior.coords, holes=smoothed_holes)

            walls.append(full_poly)  # type: ignore

        return walls

    def to_namo_world(self, robot_radius_cm: float = 60) -> World:
        map_cell_size = robot_radius_cm * 2.5
        world_width = self.width * map_cell_size
        world_height = self.height * map_cell_size
        world_cell_size = round(robot_radius_cm / 2.0)
        config = NamoConfigModel(cell_size_cm=world_cell_size * 100, agents=[])
        walls = self.get_wall_polygons(cell_size=map_cell_size)
        map = BinaryOccupancyGrid(
            static_polygons=walls,
            cell_size=world_cell_size,
            width=world_width,
            height=world_height,
        )
        world = World(
            map=map,
            collision_margin=map_cell_size,
            generate_report=config.generate_report,
            random_seed=config.random_seed,
            svg_config=config,
        )
        return world
