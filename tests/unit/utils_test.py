import unittest

from shapely.geometry import Polygon

from namosim.utils import utils
from namosim.world import binary_occupancy_grid


class GraphSearchTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_polygon_to_grid(self):
        bounds_polygon = Polygon(
            [(-12.9, -11.9), (-12.9, 11.9), (12.9, 11.9), (12.9, -11.9)]
        )
        test_polygon = Polygon([(6.1, 5.1), (6.1, 6.1), (8.1, 5.1)])
        # narrow_polygon
        inflation_radius = 2.0
        inflated_test_polygon = test_polygon.buffer(inflation_radius)
        res = 1.0

        grid_params = binary_occupancy_grid.grid_parameters(
            [bounds_polygon, test_polygon], res
        )

        (
            reference_subgrid,
            reference_subgrid_min_d_x,
            reference_subgrid_min_d_y,
        ) = utils.reference_polygon_to_subgrid(
            inflated_test_polygon, res, grid_params.grid_pose, fill=True
        )
        reference_cells_set = utils.reference_subgrid_to_cells_set(reference_subgrid)

        (
            projected_polygon,
            d_width,
            d_height,
            min_d_x,
            min_d_y,
        ) = utils.polygon_to_subgrid_polygon_and_parameters(
            inflated_test_polygon, res, grid_params.grid_pose
        )
        custom_cells_set = utils.accurate_rasterize_to_cells(
            projected_polygon, d_width, d_height, res, fill=True
        )
        utils.accurate_rasterize_to_subgrid(
            projected_polygon, d_width, d_height, res, fill=True
        )

        self.assertEqual(custom_cells_set, reference_cells_set)

        res_2 = 0.1
        basic_polygons = {
            1: Polygon(
                [
                    (-0.022476533175288527, 2.077879660203765),
                    (-2.142960375920752, 2.077879660203765),
                    (-2.142960375920752, 1.9202271893432674),
                    (-0.022476533175288527, 1.9202271893432674),
                    (-0.022476533175288527, 2.077879660203765),
                    (-0.022476533175288527, 2.077879660203765),
                ]
            ),
            2: Polygon(
                [
                    (2.1428570823808313, 2.0778372872745745),
                    (1.0374827138543168, 2.0778372872745745),
                    (1.0374827138543168, 1.9201848107696216),
                    (2.1428570823808313, 1.9201848107696216),
                    (2.1428570823808313, 2.0778372872745745),
                    (2.1428570823808313, 2.0778372872745745),
                ]
            ),
            3: Polygon(
                [
                    (-0.5066130443371999, 0.8858727457454914),
                    (-1.9780165636553495, 0.8858727457454914),
                    (-1.9780165636553495, -1.894608698106285),
                    (1.9701621369909406, -1.894608698106285),
                    (1.9701621369909406, 0.8963453560240451),
                    (0.4987587559619566, 0.8963453560240451),
                    (0.4987587559619566, 1.0534347077583042),
                    (2.142960375920752, 1.0534347077583042),
                    (2.142960375920752, -2.077879660203765),
                    (-2.1351059069229246, -2.077879660203765),
                    (-2.1351059069229246, 1.058671012897581),
                    (-0.5069306860835938, 1.058671012897581),
                    (-0.5066130443371999, 0.8858727457454914),
                    (-0.5066130443371999, 0.8858727457454914),
                ]
            ),
            4: Polygon(
                [
                    (-0.5248448339118901, 1.7772249033386955),
                    (-1.053948522563712, 1.7772249033386955),
                    (-1.053948522563712, 1.248134225157339),
                    (-0.5248448339118901, 1.248134225157339),
                    (-0.5248448339118901, 1.7772249033386955),
                    (-0.5248448339118901, 1.7772249033386955),
                ]
            ),
        }
        inflated_basic_polygons = {
            uid: p.buffer(res_2) for uid, p in basic_polygons.items()
        }
        basic_grid_params = binary_occupancy_grid.grid_parameters(
            basic_polygons.values(), res_2
        )

        (
            basic_polygon_3_projected,
            d_width,
            d_height,
            min_d_x,
            min_d_y,
        ) = utils.polygon_to_subgrid_polygon_and_parameters(
            inflated_basic_polygons[3], res_2, grid_params.grid_pose
        )
        # basic_polygon_3_custom_cells_set = utils.accurate_rasterize_to_cells(basic_polygon_3_projected, d_width, d_height, res_2, fill=True)
        utils.accurate_rasterize_to_subgrid(
            basic_polygon_3_projected, d_width, d_height, res_2, fill=True
        )
        (
            basic_polygon_3_ref_subgrid,
            basic_polygon_3_ref_subgrid_min_d_x,
            basic_polygon_3_ref_subgrid_min_d_y,
        ) = utils.reference_polygon_to_subgrid(
            inflated_basic_polygons[3], res_2, basic_grid_params.grid_pose, fill=True
        )

        print()


if __name__ == "__main__":
    unittest.main()
