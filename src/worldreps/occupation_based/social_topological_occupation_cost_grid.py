import skimage.morphology as skimage_morph
import scipy.ndimage.morphology as scipy_morph
import matplotlib.pyplot as plt
import numpy as np
import copy
import src.utils
# from skimage.morphology import medial_axis
# import cv2
from src.display.ros_publisher import RosPublisher


class SocialTopologicalOccupationCostGrid:
    """
    Resources to do skeletonization through Voronoi's algorithm with Shapely + Scipy:
    https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.Voronoi.html
    https://pypi.org/project/geovoronoi/
    https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons
    """
    def __init__(self):
        # TODO Properly parameterize all this...
        self.half_1_u_p = 0.45
        self.half_2_u_p = 0.70
        self.half_3_u_p = 0.90
        self.half_4_u_p = 1.20

        self.cost_value_at_0_u_p = 0.0
        self.cost_value_before_1_u_p = 0.1
        self.cost_value_at_1_u_p = 1.0
        self.cost_value_at_2_u_p = 0.9
        self.cost_value_at_3_u_p = 0.75
        self.cost_value_at_4_u_p_and_beyond = 0.25

        self.curve_0_to_1_u_p = (self.cost_value_before_1_u_p - self.cost_value_at_0_u_p) / (self.half_1_u_p - 0.0)
        self.offset_0_to_1_u_p = self.cost_value_before_1_u_p - self.curve_0_to_1_u_p * self.half_1_u_p

        self.curve_1_to_2_u_p = (self.cost_value_at_2_u_p - self.cost_value_at_1_u_p) / (
                    self.half_2_u_p - self.half_1_u_p)
        self.offset_1_to_2_u_p = self.cost_value_at_2_u_p - self.curve_1_to_2_u_p * self.half_2_u_p

        self.curve_2_to_3_u_p = (self.cost_value_at_3_u_p - self.cost_value_at_2_u_p) / (
                    self.half_3_u_p - self.half_2_u_p)
        self.offset_2_to_3_u_p = self.cost_value_at_3_u_p - self.curve_2_to_3_u_p * self.half_3_u_p

        self.curve_3_to_4_u_p = (self.cost_value_at_4_u_p_and_beyond - self.cost_value_at_3_u_p) / (
                    self.half_4_u_p - self.half_3_u_p)
        self.offset_3_to_4_u_p = self.cost_value_at_4_u_p_and_beyond - self.curve_3_to_4_u_p * self.half_4_u_p

        self.decay_factor = 0.02
        self.keep_number_of_decimals = 10
        self.decimals_multiplicator = 10 ** self.keep_number_of_decimals
        self.decay_limit = self.decimals_multiplicator * self.cost_value_at_4_u_p_and_beyond

        self.rp = RosPublisher()

    def _compute_social_costmap_for_entities(self, world, entities_uids, restrict_4_neighbors=True):
        # Acceptable transitions from current grid element to neighbors
        neighborhood_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighborhood_8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        if restrict_4_neighbors:
            neighborhood = neighborhood_4
        else:
            neighborhood = neighborhood_8

        # TODO :
        #  - Add support for restrict_4_neighbors
        #  - Add loop for building the final_array from the skeleton values
        world_copy_without_entities = copy.deepcopy(world)
        world_copy_without_entities.remove_entities(entities_uids)

        grid_without_entities = world_copy_without_entities.get_grid()
        plt.imshow(grid_without_entities); plt.show()
        booleanized_grid = np.zeros(grid_without_entities.shape, dtype=np.bool)
        booleanized_grid[grid_without_entities == 0] = True
        plt.imshow(booleanized_grid); plt.show()

        # Distance transform
        test_distance_transform = scipy_morph.distance_transform_cdt(booleanized_grid, 'chessboard')
        # test_distance_transform = scipy_morph.distance_transform_edt(booleanized_grid)
        plt.imshow(test_distance_transform); plt.show()

        # Skeleton
        test_skeleton = skimage_morph.skeletonize(booleanized_grid)
        # test_skeleton = medial_axis(booleanized_grid, return_distance=True)[0]
        plt.imshow(test_skeleton); plt.show()

        skeleton_cells_arrays = np.where(test_skeleton == True)
        final_array = np.full(test_distance_transform.shape, -1)
        width, height = final_array.shape[0], final_array.shape[1]
        skeleton_cells_nb = len(skeleton_cells_arrays[0])
        closed_cell_set = set()

        ordered_value_list = []

        for i in range(skeleton_cells_nb):
            x, y = skeleton_cells_arrays[0][i], skeleton_cells_arrays[1][i]
            closed_cell_set.add((x, y))
            value = int(self.skeleteton_social_cost_function(world, test_distance_transform[x][y]) * self.decimals_multiplicator)
            final_array[x][y] = value
            if value not in ordered_value_list:
                ordered_value_list.append(value)
                ordered_value_list.sort()


        # Min variant
        cur_set = closed_cell_set
        prev_set = cur_set
        while cur_set:
            self.rp.publish_grid_map(final_array / float(self.decimals_multiplicator / 100), world.dd)
            # plt.imshow(final_array); plt.show()
            next_set = set()
            for current in cur_set:
                for i, j in neighborhood_8:
                    neighbor = current[0] + i, current[1] + j
                    if neighbor not in cur_set and neighbor not in prev_set:
                        # Check that neighbor exists within the map
                        if src.utils.utils.is_in_matrix(neighbor, width, height) and booleanized_grid[neighbor[0]][neighbor[1]]:
                            # MIN CASE
                            _min = float("inf")
                            # # AVG CASE
                            # _avg = 0
                            # _count = 0
                            for k, l in neighborhood_8:
                                neighbor_of_neighbor = neighbor[0] + k, neighbor[1] + l
                                if src.utils.utils.is_in_matrix(neighbor_of_neighbor, width, height):
                                    n_o_n_value = final_array[neighbor_of_neighbor[0]][neighbor_of_neighbor[1]]
                                    # print("Value at current cell {cell} : {val}".format(
                                    #     cell=str(current), val=final_array[current[0]][current[1]]))
                                    # print("Value at neighbor cell {cell} : {val}".format(
                                    #         cell=str(neighbor), val=final_array[neighbor[0]][neighbor[1]]))
                                    # print("Value at neighbor of neighbor cell {cell} : {val}".format(
                                    #     cell=str(neighbor_of_neighbor), val=n_o_n_value))
                                    # print("----------------------------------------------------------------------")
                                    # MIN CASE
                                    if neighbor_of_neighbor not in next_set:
                                        if n_o_n_value != -1:
                                            if n_o_n_value < _min:
                                                _min = n_o_n_value
                                    # # AVG CASE
                                    # if neighbor_of_neighbor not in next_set:
                                    #     if n_o_n_value != -1:
                                    #         _avg += n_o_n_value
                                    #         _count += 1
                            # MIN CASE
                            final_array[neighbor[0]][neighbor[1]] = self.decay_function(_min)
                            # # AVG CASE
                            # _avg = int(float(_avg) / float(_count))
                            # final_array[neighbor[0]][neighbor[1]] = self.decay_function(_avg)
                            next_set.add(neighbor)
            prev_set = cur_set
            cur_set = next_set

        self.rp.publish_grid_map(final_array / float(self.decimals_multiplicator / 100), world.dd)

        final_array = final_array / float(self.decimals_multiplicator)
        return final_array

    def skeleteton_social_cost_function(self, world, dist_in_cells):
        dist_real = dist_in_cells * world.dd.res

        if 0.0 < dist_real < self.half_1_u_p:
            return self.curve_0_to_1_u_p * dist_real + self.offset_0_to_1_u_p
        elif self.half_1_u_p <= dist_real < self.half_2_u_p:
            return self.curve_1_to_2_u_p * dist_real + self.offset_1_to_2_u_p
        elif self.half_2_u_p <= dist_real < self.half_3_u_p:
            return self.curve_2_to_3_u_p * dist_real + self.offset_2_to_3_u_p
        elif self.half_3_u_p <= dist_real < self.half_4_u_p:
            return self.curve_3_to_4_u_p * dist_real + self.offset_3_to_4_u_p
        elif self.half_4_u_p <= dist_real:
            return self.cost_value_at_4_u_p_and_beyond
        else:
            return -1.0

    def decay_function(self, cost):
        return cost - cost * self.decay_factor
