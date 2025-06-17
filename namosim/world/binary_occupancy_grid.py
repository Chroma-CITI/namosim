import math
import os
import typing as t
from collections import deque

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
from typing_extensions import Self

from namosim.data_models import (
    GridCellModel,
    GridCellSet,
    MapYamlConfigModel,
    Pose2D,
)
from namosim.utils import utils


class BinaryOccupancyGrid:
    def __init__(
        self,
        *,
        cell_size: float,
        width: float,
        height: float,
        grid: npt.NDArray[np.bool_] | None = None,
        static_polygons: t.List[Polygon] | None = None,
        inflation_radius: float = 0.0,
        grid_pose: Pose2D = Pose2D(0, 0, 0),
        neighborhood: t.Sequence[GridCellModel] = utils.CHESSBOARD_NEIGHBORHOOD,
    ):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.d_width = math.ceil(width / cell_size)
        self.d_height = math.ceil(height / cell_size)
        self.inflation_radius = inflation_radius
        self.grid_pose = grid_pose
        self.neighborhood = neighborhood
        self.cell_sets: t.Dict[str, GridCellSet] = dict()
        self.static_cells: GridCellSet = set()
        self.deactivated_entities_cell_sets = {}
        self.aabb_polygon = box(*self.get_bounds())

        if grid is not None:
            self.set_grid(grid)
        else:
            self.grid = np.zeros((self.d_width, self.d_height), dtype=np.int16)

        if static_polygons is not None:
            self.reset_static_polygons(static_polygons=static_polygons)

    def find_nearest_free_cell(self, cell: GridCellModel) -> GridCellModel | None:
        # do BFS to find the nearest unoccupied cell
        frontier = deque([cell])
        visited = set()
        while len(frontier) > 0:
            x = frontier.popleft()
            visited.add(x)
            if self.get_cell_value(x) == 0:
                return x
            for n in self.get_neighbors(x):
                if n in visited:
                    continue
                frontier.append(n)
        return None

    def get_cell_value(self, cell: GridCellModel) -> int:
        return self.grid[cell[0]][cell[1]]

    def set_grid(self, grid: npt.NDArray[np.bool_]):
        self.d_width = grid.shape[0]
        self.d_height = grid.shape[1]
        self.width = self.d_width * self.cell_size
        self.height = self.d_height * self.cell_size
        x_coords, y_coords = np.where(grid == True)
        occupied_cells = set(zip(x_coords, y_coords))
        self.static_cells = occupied_cells
        self.grid = grid.astype(np.int16)

    def inflate_map_destructive(self, radius: float):
        assert len(self.cell_sets) == 0
        assert len(self.deactivated_entities_cell_sets) == 0
        assert self.inflation_radius == 0
        self.inflation_radius = radius
        self._dilate_grid(radius)
        return self

    def get_bounds(self):
        return (
            self.grid_pose[0],
            self.grid_pose[1],
            self.grid_pose[0] + self.width,
            self.grid_pose[1] + self.height,
        )

    def rasterize_polygon(
        self, polygon: Polygon, fill: bool = True
    ) -> t.Set[GridCellModel]:
        """Uses PIL to rasterize a polygon into the occupancy grid. We use PIL for this because it is
        significantly faster than a naive python implementation.
        """
        img = Image.new("L", (self.d_width, self.d_height), 0)
        poly_coordinates_in_image = [
            self.pose_to_cell(x, y) for (x, y) in polygon.exterior.coords
        ]
        ImageDraw.Draw(img).polygon(
            poly_coordinates_in_image, outline=1, fill=1 if fill else 0
        )
        subgrid = np.transpose(np.array(img, dtype=np.uint8))  # (y, x) -> (x, y)
        x_coords, y_coords = np.where(subgrid == 1)

        return set(zip(x_coords, y_coords))

    def _dilate_grid(self, radius: float):
        radius /= self.cell_size
        # Create a circular kernel with the specified radius
        grid = (self.grid != 0).astype(np.uint8)
        kernel_size = (int(2 * radius + 1), int(2 * radius + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # Apply dilation
        dilated_grid = cv2.dilate(grid, kernel, iterations=1)
        self.set_grid(np.array(dilated_grid))  # type: ignore

    def reset_static_polygons(
        self,
        static_polygons: t.List[Polygon],
    ):
        self.static_cells = set()
        self.grid *= 0
        for polygon in static_polygons:
            cells = self.rasterize_polygon(polygon.buffer(self.inflation_radius))  # type: ignore
            self.static_cells = self.static_cells.union(cells)

        for cell in self.static_cells:
            self.grid[cell[0]][cell[1]] = 1

    def update_polygons(
        self,
        new_or_updated_polygons: t.Dict[str, Polygon] | None = None,
        removed_polygons: t.Set[str] | None = None,
    ):
        """Updates the grid based on which polygons have been added, changed, or removed.

        :param new_or_updated_polygons: _description_, defaults to None
        :type new_or_updated_polygons: t.Dict[str, Polygon] | None, optional
        :param removed_polygons: _description_, defaults to None
        :type removed_polygons: t.Dict[str, Polygon] | None, optional
        :param fill: _description_, defaults to True
        :type fill: bool, optional
        :return: _description_
        :rtype: _type_
        """
        new_or_updated_cell_sets = (
            None
            if not new_or_updated_polygons
            else {
                uid: self.rasterize_polygon(new_polygon.buffer(self.inflation_radius))  # type: ignore
                for uid, new_polygon in new_or_updated_polygons.items()
            }
        )

        return self.cell_sets_update(new_or_updated_cell_sets, removed_polygons)

    def cell_sets_update(
        self,
        new_or_updated_cell_sets: t.Dict[str, GridCellSet] | None = None,
        removed_entities: t.Set[str] | None = None,
    ) -> t.Dict[str, GridCellSet]:
        prev_cell_sets = {}

        if new_or_updated_cell_sets is not None:
            for uid, new_cells_set in new_or_updated_cell_sets.items():
                if uid in self.deactivated_entities_cell_sets:
                    self.deactivated_entities_cell_sets[uid] = new_cells_set
                else:
                    if uid in self.cell_sets:
                        prev_cells = self.cell_sets[uid]
                        for cell in prev_cells:
                            self.grid[cell[0]][cell[1]] -= 1
                        prev_cell_sets[uid] = prev_cells

                    self.cell_sets[uid] = new_cells_set
                    for cell in new_cells_set:
                        self.grid[cell[0]][cell[1]] += 1

        if removed_entities is not None:
            for uid in removed_entities:
                if uid in self.deactivated_entities_cell_sets:
                    del self.deactivated_entities_cell_sets[uid]
                elif uid in self.cell_sets:
                    prev_cells = self.cell_sets[uid]
                    del self.cell_sets[uid]
                    for cell in prev_cells:
                        self.grid[cell[0]][cell[1]] -= 1
                    prev_cell_sets[uid] = prev_cells

        return prev_cell_sets

    def deactivate_entities(self, uids: t.Iterable[str]):
        for uid in uids:
            if uid not in self.deactivated_entities_cell_sets and uid in self.cell_sets:
                self.deactivated_entities_cell_sets[uid] = self.cell_sets[uid]
                for cell in self.cell_sets[uid]:
                    self.grid[cell[0]][cell[1]] -= 1
                del self.cell_sets[uid]

    def activate_entities(self, uids: t.Iterable[str]):
        for uid in uids:
            if uid in self.deactivated_entities_cell_sets:
                self.cell_sets[uid] = self.deactivated_entities_cell_sets[uid]
                del self.deactivated_entities_cell_sets[uid]
                for cell in self.cell_sets[uid]:
                    self.grid[cell[0]][cell[1]] += 1

    def cell_to_dynamic_entity_ids(self, cell: GridCellModel) -> t.Set[str]:
        """
        If cell is contained only by one obstacle o_i, returns o_i.
        If contained by no obstacle, returns None. If contained by more than one, returns -1.
        :param cell: cell coordinates (x, y)
        :type cell: tuple(int, int)
        :return: obstacle uid or 0 or -1
        :rtype: int
        """
        if self.grid[cell[0]][cell[1]] == 0:
            return set()

        result = set()
        for uid, cell_set in self.cell_sets.items():
            if cell in cell_set:
                result.add(uid)

        if len(result) > 1:
            assert self.grid[cell[0]][cell[1]] > 1
        return result

    def obstacles_uids_in_cell(self, cell: GridCellModel):
        return {uid for uid, cell_set in self.cell_sets.items() if cell in cell_set}

    def get_cell_center(self, cell: GridCellModel):
        return (
            cell[0] * self.cell_size + self.cell_size / 2,
            cell[1] * self.cell_size + self.cell_size / 2,
        )

    def cell_to_polygon(self, cell: GridCellModel) -> Polygon:
        return box(
            minx=self.grid_pose[0] + cell[0] * self.cell_size,
            miny=self.grid_pose[1] + cell[1] * self.cell_size,
            maxx=self.grid_pose[0] + cell[0] * self.cell_size + self.cell_size,
            maxy=self.grid_pose[1] + cell[1] * self.cell_size + self.cell_size,
        )

    def get_neighbors(self, cell: GridCellModel) -> t.List[GridCellModel]:
        result: t.List[GridCellModel] = []
        ci, cj = cell
        for i, j in utils.CHESSBOARD_NEIGHBORHOOD:
            ni, nj = i + ci, j + cj
            if self.is_cell_in_bounds((ni, nj)):
                result.append((ni, nj))
        return result

    def is_cell_in_bounds(self, cell: GridCellModel) -> bool:
        if cell[0] < 0 or cell[0] >= self.grid.shape[0]:
            return False
        if cell[1] < 0 or cell[1] >= self.grid.shape[1]:
            return False
        return True

    def pose_to_cell(self, x: float, y: float) -> GridCellModel:
        """Computes the grid cell corresponding to a real-valued (x, y) position"""
        cx = int(math.floor((x - self.grid_pose[0]) / self.cell_size))
        cx = min(max(0, cx), self.d_width - 1)
        cy = int(math.floor((y - self.grid_pose[1]) / self.cell_size))
        cy = min(max(0, cy), self.d_height - 1)
        return (cx, cy)

    def polygon_has_collisions(self, polygon: Polygon) -> bool:
        """Computes the grid cell corresponding to a real-valued (x, y) position"""
        img = Image.new("L", (self.d_width, self.d_height), 0)
        poly_coordinates_in_image = [
            (x / self.cell_size, y / self.cell_size)
            for (x, y) in polygon.exterior.coords
        ]
        ImageDraw.Draw(img).polygon(poly_coordinates_in_image, outline=1, fill=1)
        polygon_grid = np.transpose(np.array(img, dtype=np.uint8))  # (y, x) -> (x, y)
        grid_map = self.grid != 0
        overlap = np.any(grid_map * polygon_grid > 0)
        return overlap  # type: ignore

    def to_image(self) -> Image.Image:
        # grid = np.flipud(self.grid)
        grid = self.grid.astype(np.float32)
        grid = np.transpose(grid)  # (x, y) -> (y, x)
        grid = np.flipud(grid)
        grid = np.where(grid == 0, 1, 0)
        alpha = np.where(grid == 0, 255, 0).astype(np.uint8)
        grid *= 255
        grid = grid.astype(np.uint8)

        h, w = grid.shape
        rgba = (np.ones((4, h, w)) * grid).astype(np.uint8)
        rgba[3] = alpha
        rgba = np.transpose(rgba, (1, 2, 0))
        return Image.fromarray(rgba, "RGBA")

    @classmethod
    def load_from_yaml(
        cls,
        yaml_file: str,
    ) -> Self:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)

        config = MapYamlConfigModel(**data)
        map_image_path = os.path.join(os.path.dirname(yaml_file), config.image)
        map_image = Image.open(map_image_path)
        map_image = map_image.convert("L")  # Convert to grayscale
        orig = np.array(map_image)
        grid = np.array(map_image)

        free_threshold = 220

        # Apply thresholds from the YAML file
        grid[orig <= free_threshold] = 1
        grid[orig > free_threshold] = 0

        grid = np.flipud(grid)
        grid = grid.transpose()

        grid = grid.astype(np.bool_)
        if config.negate:
            grid = np.logical_not(grid)

        d_width, d_height = grid.shape
        width = d_width * config.resolution
        height = d_height * config.resolution
        return cls(cell_size=config.resolution, width=width, height=height, grid=grid)

    def draw_polygon_on_map(
        self,
        polygon: Polygon,
        polygon_color: tuple = (255, 0, 0),
    ) -> Image.Image:
        """
        Draw a Shapely polygon (coordinates in meters) on a Pillow image.

        Args:
            polygon (Polygon): Shapely Polygon with coordinates in meters.
            image_size (tuple): Width and height of the output image in pixels (default: (500, 500)).
            polygon_color (tuple): RGB color for the polygon (default: red).
            background_color (tuple): RGB color for the background (default: white).

        Returns:
            Image.Image: Pillow image with the drawn polygon.
        """
        # Create a new blank image
        image = self.to_image()
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        draw = ImageDraw.Draw(image)

        poly_coordinates_in_image = [
            self.pose_to_cell(x, y) for (x, y) in polygon.exterior.coords
        ]

        # Draw the polygon
        draw.polygon(poly_coordinates_in_image, fill=polygon_color, outline=(0, 0, 0))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image
