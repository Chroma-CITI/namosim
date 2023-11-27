import math
import typing as t

import numpy as np
from shapely.geometry import Polygon

from namosim.models import GridCellModel, GridCellSet, PoseModel
from namosim.utils import utils


class GridParams:
    """
    Represents the position of an entity on the grid, including its pose,
    grid coordinates, real coordinates, and axis-aligned bounding box.
    """

    def __init__(
        self,
        pose: PoseModel,
        d_width: int,
        d_height: int,
        r_width: float,
        r_height: float,
        aabb_polygon: Polygon,
    ):
        """
        :param pose: The entity's pose in real coordinates
        :type PoseModel

        :param d_width: Grid-space width of the entity's bounding box
        :type int

        :param d_height: Grid-space height of the entity's bounding box
        :type int

        :param r_width: Real-space width of the entity's bounding box
        :type float

        :param r_height: Real-space height of the entity's bounding box
        :type float

        :param aabb_polygon: The entity's axis-aligned bounding box in real coordinates
        :type Polygon

        :return: A `GridParams` object
        :rtype: GridParams
        """
        (
            self.grid_pose,
            self.d_width,
            self.d_height,
            self.r_width,
            self.r_height,
            self.aabb_polygon,
        ) = (pose, d_width, d_height, r_width, r_height, aabb_polygon)

    def __eq__(self, other: object):
        if isinstance(other, GridParams):
            return (
                self.grid_pose,
                self.d_width,
                self.d_height,
                self.r_width,
                self.r_height,
                self.aabb_polygon,
            ) == (
                other.grid_pose,
                other.d_width,
                other.d_height,
                other.r_width,
                other.r_height,
                other.aabb_polygon,
            )
        return False

    def all(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return (
            self.grid_pose,
            self.d_width,
            self.d_height,
            self.r_width,
            self.r_height,
            self.aabb_polygon,
        )


def grid_parameters(polygons: t.Iterable[Polygon], res: float):
    r_min_x, r_min_y, r_max_x, r_max_y = utils.map_bounds(polygons)
    min_x, min_y = math.floor(r_min_x / res) * res, math.floor(r_min_y / res) * res
    max_x, max_y = math.ceil(r_max_x / res) * res, math.ceil(r_max_y / res) * res
    d_width = abs(int(math.floor(r_min_x / res))) + abs(int(math.ceil(r_max_x / res)))
    d_height = abs(int(math.floor(r_min_y / res))) + abs(int(math.ceil(r_max_y / res)))
    real_width, real_height = d_width * res, d_height * res
    real_pose = min_x, min_y, 0.0
    aabb_polygon = Polygon(
        [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    )
    return GridParams(
        pose=real_pose,
        d_width=d_width,
        d_height=d_height,
        r_width=real_width,
        r_height=real_height,
        aabb_polygon=aabb_polygon,
    )


class BinaryOccupancyGrid:
    def __init__(
        self,
        polygons: t.Dict[int, Polygon],
        res: float,
        neighborhood: t.Sequence[GridCellModel] = utils.CHESSBOARD_NEIGHBORHOOD,
        params: GridParams | None = None,
        fill: bool = True,
    ):
        """_summary_

        :param polygons: a dictionary mapping entity uids to polygons
        :type polygons: t.Dict[int, Polygon]
        :param res: the grid resolution parameter
        :type res: float
        :param neighborhood: _description_, defaults to utils.CHESSBOARD_NEIGHBORHOOD
        :type neighborhood: t.Sequence[GridCellModel] , optional
        :param params: _description_, defaults to None
        :type params: `GridParams`, optional
        :param fill: whether to fill grid cells lying within polygons, defaults to True
        :type fill: bool, optional
        """
        self.res = res

        self.params = params if params else grid_parameters(polygons.values(), res)

        (
            self.grid_pose,
            self.d_width,
            self.d_height,
            self.r_width,
            self.r_height,
            self.aabb_polygon,
        ) = self.params.all()

        self.neighborhood = neighborhood

        self.cells_sets: t.Dict[int, GridCellSet] = dict()
        self.grid = np.zeros((self.d_width, self.d_height), dtype=np.int16)

        self.deactivated_entities_cells_sets = {}

        self.update(new_or_updated_polygons=polygons, fill=fill)

    def update(
        self,
        new_or_updated_polygons: t.Dict[int, Polygon] | None = None,
        removed_polygons: t.Set[int] | None = None,
        fill: bool = True,
    ):
        """Updates the grid based on which polygons have been added, changed, or removed.

        :param new_or_updated_polygons: _description_, defaults to None
        :type new_or_updated_polygons: t.Dict[int, Polygon] | None, optional
        :param removed_polygons: _description_, defaults to None
        :type removed_polygons: t.Dict[int, Polygon] | None, optional
        :param fill: _description_, defaults to True
        :type fill: bool, optional
        :return: _description_
        :rtype: _type_
        """
        new_or_updated_cells_sets = (
            None
            if not new_or_updated_polygons
            else {
                uid: utils.accurate_rasterize_in_grid(
                    new_polygon,
                    self.res,
                    self.grid_pose,
                    self.d_width,
                    self.d_height,
                    fill=fill,
                )
                for uid, new_polygon in new_or_updated_polygons.items()
            }
        )

        return self.cells_sets_update(new_or_updated_cells_sets, removed_polygons)

    def cells_sets_update(
        self,
        new_or_updated_cells_sets: t.Dict[int, GridCellSet] | None = None,
        removed_entities: t.Set[int] | None = None,
    ) -> t.Dict[int, GridCellSet]:
        prev_cells_sets = {}

        if new_or_updated_cells_sets is not None:
            for uid, new_cells_set in new_or_updated_cells_sets.items():
                if uid in self.deactivated_entities_cells_sets:
                    self.deactivated_entities_cells_sets[uid] = new_cells_set
                else:
                    if uid in self.cells_sets:
                        prev_cells = self.cells_sets[uid]
                        for cell in prev_cells:
                            self.grid[cell[0]][cell[1]] -= 1
                        prev_cells_sets[uid] = prev_cells

                    self.cells_sets[uid] = new_cells_set
                    for cell in new_cells_set:
                        self.grid[cell[0]][cell[1]] += 1

        if removed_entities is not None:
            for uid in removed_entities:
                if uid in self.deactivated_entities_cells_sets:
                    del self.deactivated_entities_cells_sets[uid]
                else:
                    prev_cells = self.cells_sets[uid]
                    del self.cells_sets[uid]
                    for cell in prev_cells:
                        self.grid[cell[0]][cell[1]] -= 1
                    prev_cells_sets[uid] = prev_cells

        return prev_cells_sets

    def deactivate_entities(self, uids: t.Iterable[int]):
        for uid in uids:
            if (
                uid not in self.deactivated_entities_cells_sets
                and uid in self.cells_sets
            ):
                self.deactivated_entities_cells_sets[uid] = self.cells_sets[uid]
                for cell in self.cells_sets[uid]:
                    self.grid[cell[0]][cell[1]] -= 1
                del self.cells_sets[uid]

    def activate_entities(self, uids: t.Iterable[int]):
        for uid in uids:
            if uid in self.deactivated_entities_cells_sets:
                self.cells_sets[uid] = self.deactivated_entities_cells_sets[uid]
                del self.deactivated_entities_cells_sets[uid]
                for cell in self.cells_sets[uid]:
                    self.grid[cell[0]][cell[1]] += 1

    def only_obstacle_uid_in_cell(self, cell: GridCellModel):
        """
        If cell is contained only by one obstacle o_i, returns o_i.
        If contained by no obstacle, returns 0. If contained by more than one, returns -1.
        :param cell: cell coordinates (x, y)
        :type cell: tuple(int, int)
        :return: obstacle uid or 0 or -1
        :rtype: int
        """
        if self.grid[cell[0]][cell[1]] == 0:
            return 0
        elif self.grid[cell[0]][cell[1]] > 1:
            return -1
        else:
            for uid, cell_set in self.cells_sets.items():
                if cell in cell_set:
                    return uid
            raise RuntimeError(
                "It should be impossible for an occupied cell of the grid to not be in any cells set."
            )

    def obstacles_uids_in_cell(self, cell: GridCellModel):
        return {uid for uid, cell_set in self.cells_sets.items() if cell in cell_set}


class BinaryInflatedOccupancyGrid(BinaryOccupancyGrid):
    """
    Represents an occupancy grid in which each polygon has been inflated the robot's radius
    to avoid collision.
    """

    def __init__(
        self,
        polygons: t.Dict[int, Polygon],
        res: float,
        inflation_radius: float,
        neighborhood: t.Sequence[GridCellModel] = utils.CHESSBOARD_NEIGHBORHOOD,
        params: GridParams | None = None,
        fill: bool = True,
    ):
        self.inflation_radius = inflation_radius
        super().__init__(polygons, res, neighborhood, params, fill)

    def update(
        self,
        new_or_updated_polygons: t.Dict[int, Polygon] | None = None,
        removed_polygons: t.Set[int] | None = None,
        fill: bool = True,
    ):
        if new_or_updated_polygons:
            inflated_polygons = {
                uid: polygon.buffer(self.inflation_radius)
                for uid, polygon in new_or_updated_polygons.items()
            }
            return BinaryOccupancyGrid.update(
                self, inflated_polygons, removed_polygons, fill=fill
            )
        else:
            return BinaryOccupancyGrid.update(
                self, new_or_updated_polygons, removed_polygons
            )
