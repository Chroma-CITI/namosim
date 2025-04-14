from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely.geometry import Polygon
import numpy as np
from nav_msgs.msg import OccupancyGrid


def test_grid():
    grid = BinaryOccupancyGrid(width=3, height=3, cell_size=1)
    assert grid.grid.shape == (3, 3)
    assert grid.get_bounds() == (0, 0, 3, 3)
    assert grid.grid.sum() == 0
    triangle = Polygon([(0.5, 1.5), (2.5, 1.5), (1.5, 0.5), (0.5, 1.5)])
    grid.update_polygons({"x": triangle})
    assert grid.cell_sets["x"] == set([(0, 1), (1, 1), (2, 1), (1, 0)])
    assert np.allclose(grid.grid, np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]).T)


def test_inflation_radius():
    grid = BinaryOccupancyGrid(width=3, height=3, cell_size=1, inflation_radius=0.8)
    triangle = Polygon([(0.5, 1.5), (2.5, 1.5), (1.5, 0.5), (0.5, 1.5)])
    grid.update_polygons({"x": triangle})
    assert grid.cell_sets["x"] == set(
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    )


def test_init_from_grid():
    grid = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.bool_)

    bin = BinaryOccupancyGrid(grid=grid, cell_size=1, width=3, height=3)
    assert bin.static_cells == set([(0, 1), (1, 1), (1, 2), (1, 0)])


def test_from_occupancy_grid():
    msg = OccupancyGrid()
    msg.info.resolution = 0.05  # Example resolution: 0.05 meters/cell
    msg.info.width = 3
    msg.info.height = 3
    data = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    msg.data = data.flatten().tolist()
    grid = (
        np.array(msg.data, dtype=np.int16).reshape((3, 3)) > 0
    )  # Assuming int8 is enough (-1, 0, 100)
    grid_pose = (msg.info.origin.position.x, msg.info.origin.position.y, 0)
    bin = BinaryOccupancyGrid(
        grid=grid, cell_size=1, width=3, height=3, grid_pose=grid_pose
    )
    assert bin.static_cells == set([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)])
