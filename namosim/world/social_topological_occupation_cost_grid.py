import os
import time
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scipy_morph
import skimage.morphology as skimage_morph
from matplotlib import colormaps as cm
from PIL import Image
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point

import namosim.display.ros2_publisher as rp
from namosim.utils import utils

rel_path_to_costmap_logs_dir = "../../../logs/costmaps/"
abs_path_to_costmap_logs_dir = os.path.join(
    os.path.dirname(__file__), rel_path_to_costmap_logs_dir
)


def display_or_log(
    grid,
    suffix,
    start_time_str,
    debug_display=True,
    log_costmaps=True,
    logs_dir=abs_path_to_costmap_logs_dir,
):
    if debug_display or log_costmaps:
        if grid.dtype == np.float64:
            if np.max(grid) > 1.0:
                grid = (grid - np.min(grid)) / np.ptp(grid)
            rainbow_colormap = cm.get_cmap("gist_rainbow_r")  # type: ignore
            color_grid = rainbow_colormap(grid)
            color_grid_int = np.uint8(color_grid * 255)
            red, green, blue = (
                color_grid_int[:, :, 0],  # type: ignore
                color_grid_int[:, :, 1],  # type: ignore
                color_grid_int[:, :, 2],  # type: ignore
            )
            mask = (red == 255) & (green == 0) & (blue == 191)
            color_grid_int[:, :, :4][mask] = [0, 0, 0, 0]  # type: ignore
            img = Image.fromarray(color_grid_int).convert("RGBA")
        else:
            img = Image.fromarray(grid).convert("RGB")
        if debug_display:
            img.show()
        if log_costmaps:
            log_filename = os.path.join(
                os.path.dirname(logs_dir), start_time_str + suffix
            )
            counter = 0
            while os.path.isfile(
                ("" if counter == 0 else ("-" + str(counter))) + log_filename + ".png"
            ):
                # or os.path.isfile(("" if counter == 0 else ("-" + str(counter))) + log_filename + ".npy")):
                counter += 1
            img.save(
                log_filename + ("" if counter == 0 else ("-" + str(counter))) + ".png"
            )
            # np.save(log_filename + ("" if counter == 0 else ("-" + str(counter))) + ".npy", grid)


def skeleteton_social_cost_function(
    dist_in_cells,
    cell_size=1.0,
    cost_value_at_0_u_p=0.0,
    cost_value_before_1_u_p=0.1,
    cost_value_at_1_u_p=1.0,
    cost_value_at_2_u_p=0.9,
    cost_value_at_3_u_p=0.75,
    cost_value_at_4_u_p_and_beyond=0.25,
):
    half_1_u_p = 0.45
    half_2_u_p = 0.70
    half_3_u_p = 0.90
    half_4_u_p = 1.20

    curve_0_to_1_u_p = (cost_value_before_1_u_p - cost_value_at_0_u_p) / (
        half_1_u_p - 0.0
    )
    offset_0_to_1_u_p = cost_value_before_1_u_p - curve_0_to_1_u_p * half_1_u_p

    curve_1_to_2_u_p = (cost_value_at_2_u_p - cost_value_at_1_u_p) / (
        half_2_u_p - half_1_u_p
    )
    offset_1_to_2_u_p = cost_value_at_2_u_p - curve_1_to_2_u_p * half_2_u_p

    curve_2_to_3_u_p = (cost_value_at_3_u_p - cost_value_at_2_u_p) / (
        half_3_u_p - half_2_u_p
    )
    offset_2_to_3_u_p = cost_value_at_3_u_p - curve_2_to_3_u_p * half_3_u_p

    curve_3_to_4_u_p = (cost_value_at_4_u_p_and_beyond - cost_value_at_3_u_p) / (
        half_4_u_p - half_3_u_p
    )
    offset_3_to_4_u_p = cost_value_at_4_u_p_and_beyond - curve_3_to_4_u_p * half_4_u_p

    dist_real = dist_in_cells * cell_size

    if 0.0 < dist_real < half_1_u_p:
        return curve_0_to_1_u_p * dist_real + offset_0_to_1_u_p
    if half_1_u_p <= dist_real < half_2_u_p:
        return curve_1_to_2_u_p * dist_real + offset_1_to_2_u_p
    if half_2_u_p <= dist_real < half_3_u_p:
        return curve_2_to_3_u_p * dist_real + offset_2_to_3_u_p
    if half_3_u_p <= dist_real < half_4_u_p:
        return curve_3_to_4_u_p * dist_real + offset_3_to_4_u_p
    if half_4_u_p <= dist_real:
        return cost_value_at_4_u_p_and_beyond
    return -1.0


def skeleteton_social_cost_function_02(dist_in_cells, cell_size=1.0):
    distances = [0.0, 0.275, 0.45, 0.70, 0.90, 1.20, 1.50]
    values = [1.0, 1.000, 0.90, 0.78, 0.7, 0.6, 0.5]

    curves, offsets = [], []
    for counter in range(len(values) - 1):
        curve = (values[counter + 1] - values[counter]) / (
            distances[counter + 1] - distances[counter]
        )
        curves.append(curve)
        offset = values[counter + 1] - curve * distances[counter + 1]
        offsets.append(offset)
        counter += 1

    dist_real = dist_in_cells * cell_size

    for counter in range(len(values) - 1):
        if distances[counter] <= dist_real < distances[counter + 1]:
            return curves[counter] * dist_real + offsets[counter]

    if distances[-1] <= dist_real:
        return values[-1]

    return -1.0


# def plot_conversion_function():
#     import matplotlib.pyplot as plt
#
#     plt.style.use("ggplot")
#
#     dist = [
#         0.0, 0.2, 0.45, 0.70, 0.90, 1.20, 1.50, 2.]
#     social_cost = [
#         1.0, 1.0, 0.90, 0.80, 0.70, 0.50, 0.25, 0.25]
#     plt.xticks(
#         dist, ['0.0', 'Rhuman', '1HUP', '2HUP', '3HUP', '4HUP', '5HUP', '']
#     )
#     plt.plot(dist, social_cost)
#     plt.xlabel("Original Space Allowance [m]")
#     plt.ylabel("Social Occupation Cost")
#     plt.title("Social Occupation Cost as a function of the Original Space Allowance", wrap=True)
#     plt.grid(True)
#
#     # plt.show()
#
#     # import tikzplotlib
#
#     tikzplotlib.save("test.tex")


def adaptive_lambda(v_skel_min: float, v_skel_max: float, max_dist_from_skell: float):
    return (v_skel_min / v_skel_max) ** max_dist_from_skell


def exp_decay_function(cost: float, decay_factor: float):
    return cost * decay_factor


def chessboard_distance_function(grid):
    return scipy_morph.distance_transform_cdt(grid, "chessboard")


def taxicab_distance_function(grid):
    return scipy_morph.distance_transform_cdt(grid, "taxicab")


def euclidean_distance_function(grid):
    return scipy_morph.distance_transform_edt(grid)


def skimage_skeletonize(grid):
    return skimage_morph.skeletonize(grid)


def skimage_medial_axis(grid):
    return skimage_morph.medial_axis(grid, return_distance=True)[0]


def voronoi_skeleton(entities, entities_to_ignore=tuple()):
    all_points = []
    all_polygons = []
    for entity_uid, entity in entities.items():
        if entity_uid not in entities_to_ignore:
            all_points += list(entity.polygon.exterior.coords)
            all_polygons.append(entity.polygon)
    voronoi = Voronoi(all_points)

    voronoi_vertices_and_points = [
        (tuple(vertex), Point(vertex)) for vertex in voronoi.vertices
    ]

    voronoi_graph_vertices_and_points = {}
    for vertex, point in voronoi_vertices_and_points:
        intersects = False
        for polygon in all_polygons:
            if point.intersects(polygon):
                intersects = True
                break
        if not intersects and -15.0 <= vertex[0] <= 15.0 and -15.0 <= vertex[1] <= 15.0:
            voronoi_graph_vertices_and_points[vertex] = point

    linestrings = []
    for ridge in voronoi.ridge_vertices:
        vertex_a, vertex_b = (
            tuple(voronoi.vertices[ridge[0]]),
            tuple(voronoi.vertices[ridge[1]]),
        )
        is_ridge_valid = (
            vertex_a in voronoi_graph_vertices_and_points
            and vertex_b in voronoi_graph_vertices_and_points
        )
        if is_ridge_valid:
            linestring = LineString(
                [
                    voronoi_graph_vertices_and_points[vertex_a],
                    voronoi_graph_vertices_and_points[vertex_b],
                ]
            )
            linestrings.append(linestring)
            plt.plot(*linestring.xy)
    plt.show()


def compute_social_costmap(
    *,
    binary_occ_grid,
    cell_size,
    agent_id: str,
    ros_publisher: t.Optional["rp.RosPublisher"] = None,
    neighborhood=utils.TAXI_NEIGHBORHOOD,
    skeleton_function=skeleteton_social_cost_function_02,
    decay_function=exp_decay_function,
    decay_factor=0.95,
    activate_adaptive_lambda=False,
    distance_transform_function=euclidean_distance_function,
    skeleton_transform_function=skimage_morph.thin,  # skimage_skeletonize,
    debug_display=False,
    log_costmaps=True,
    logs_dir: str = abs_path_to_costmap_logs_dir,
    skeleton_filepath=None,
):  # rel_path_to_costmap_logs_dir + "citi_saved_skeleton.png"):
    start_time_str = time.strftime("%Y-%m-%d-%Hh%Mm%Ss")

    # Transform binary occupation grid made of integers into booleans for scipy functions to work properly
    booleanized_grid = np.zeros(binary_occ_grid.shape, dtype=bool)
    booleanized_grid[binary_occ_grid == 0] = True
    display_or_log(
        booleanized_grid,
        "-occupation_grid",
        start_time_str,
        debug_display,
        log_costmaps,
        logs_dir,
    )

    # Apply distance transform to booleanized grid
    distance_transformed_grid = distance_transform_function(booleanized_grid)
    display_or_log(
        distance_transformed_grid,
        "-distance_transform",
        start_time_str,
        debug_display,
        log_costmaps,
        logs_dir,
    )

    # Apply skeleton transform to booleanized grid
    if skeleton_filepath is None:
        skeletonized_grid = skeleton_transform_function(booleanized_grid)
    else:
        skeletonized_grid = np.array(Image.open(skeleton_filepath).convert("1"))

    display_or_log(
        skeletonized_grid,
        "-skeleton",
        start_time_str,
        debug_display,
        log_costmaps,
        logs_dir,
    )

    # Extract the skeleton cells into two arrays of coordinates
    skeleton_cells_xy = np.where(skeletonized_grid)

    final_array = np.full(distance_transformed_grid.shape, -1.0)
    width, height = final_array.shape[0], final_array.shape[1]
    skeleton_cells_nb = len(skeleton_cells_xy[0])
    skeleton_cell_set = set()
    skeleton_values = []
    for i in range(skeleton_cells_nb):
        x, y = skeleton_cells_xy[0][i], skeleton_cells_xy[1][i]
        value = skeleton_function(distance_transformed_grid[x][y], cell_size)
        if value != -1.0:
            skeleton_cell_set.add((x, y))
            final_array[x][y] = value
            skeleton_values.append(value)

    display_or_log(
        final_array,
        "-skeleton_values",
        start_time_str,
        debug_display,
        log_costmaps,
        logs_dir,
    )

    # Compute adaptive lambda decay
    if activate_adaptive_lambda:
        decay_factor = adaptive_lambda(
            min(skeleton_values),
            max(skeleton_values),
            distance_transformed_grid.max() * cell_size,
        )

    if ros_publisher:
        ros_publisher.publish_social_costmap(final_array, cell_size, agent_id=agent_id)
    # time.sleep(3.0)

    cur_set = skeleton_cell_set
    prev_set = cur_set
    while cur_set:
        if debug_display:
            import matplotlib.pyplot as plt

            plt.imshow(final_array)
            plt.show()

        # X, Y = np.meshgrid(range(final_array.shape[0]), range(final_array.shape[1]))
        # Z = final_array
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        #
        # # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # surf = ax.plot_surface(X.T, Y.T, Z, cmap=cm.get_cmap("gist_rainbow_r"), linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

        next_set = set()
        for current in cur_set:
            for i, j in neighborhood:
                neighbor = current[0] + i, current[1] + j
                if neighbor not in cur_set and neighbor not in prev_set:
                    # Check that neighbor exists within the map
                    if (
                        utils.is_in_matrix(neighbor, width, height)
                        and booleanized_grid[neighbor[0]][neighbor[1]]
                    ):
                        values_in_neighborhood = []
                        # weights_in_neighborhood = []
                        for k, l in neighborhood:
                            neighbor_of_neighbor = neighbor[0] + k, neighbor[1] + l
                            if utils.is_in_matrix(neighbor_of_neighbor, width, height):
                                n_o_n_value = final_array[neighbor_of_neighbor[0]][
                                    neighbor_of_neighbor[1]
                                ]
                                if neighbor_of_neighbor not in next_set:
                                    if n_o_n_value != -1:
                                        values_in_neighborhood.append(n_o_n_value)
                                        # weights_in_neighborhood.append(1. / utils.euclidean_distance(neighbor, neighbor_of_neighbor))
                        final_array[neighbor[0]][neighbor[1]] = decay_function(
                            np.min(values_in_neighborhood), decay_factor
                        )
                        # final_array[neighbor[0]][neighbor[1]] = decay_function(
                        #     sum([values_in_neighborhood[i] * weights_in_neighborhood[i] for i in range(len(values_in_neighborhood))]) / sum(weights_in_neighborhood), decay_factor)
                        next_set.add(neighbor)
        prev_set = cur_set
        cur_set = next_set

        if ros_publisher:
            ros_publisher.publish_social_costmap(
                final_array, cell_size, agent_id=agent_id
            )

    # prev_set = skeleton_cell_set
    # cur_set = utils.get_set_neighbors_no_coll(skeleton_cell_set, binary_occ_grid, neighborhood)
    # while cur_set:
    #     # utils.matplotlib_show_grid(final_array)
    #     for cell in cur_set:
    #         valid_prev_cells = utils.get_neighbors(cell, width, height, neighborhood).intersection(prev_set)
    #         values_in_prev_cells = [
    #             final_array[valid_prev_cell[0]][valid_prev_cell[1]] for valid_prev_cell in valid_prev_cells
    #         ]
    #         final_array[cell[0]][cell[1]] = decay_function(min(values_in_prev_cells), decay_factor)
    #     next_set = utils.get_set_neighbors_no_coll(cur_set, binary_occ_grid, neighborhood, prev_set)
    #     prev_set = cur_set
    #     cur_set = next_set

    display_or_log(
        final_array,
        "-social_costmap",
        start_time_str,
        debug_display,
        log_costmaps,
        logs_dir,
    )

    # time.sleep(10.0)

    return final_array


class SocialTopologicalOccupationCostGrid:
    """
    Resources to do skeletonization through Voronoi's algorithm with Shapely + Scipy:
    https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.Voronoi.html
    https://pypi.org/project/geovoronoi/
    https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons
    """

    def __init__(
        self,
        occupation_grid,
        d_width,
        d_height,
        cell_size,
        grid_pose,
        inflation_radius,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        skeleton_function=skeleteton_social_cost_function,
        decay_function=exp_decay_function,
        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        agent_id="",
    ):
        self.occupation_grid = occupation_grid
        self.d_width, self.d_height = d_width, d_height
        self.cell_size = cell_size
        self.grid_pose = grid_pose
        self.inflation_radius = inflation_radius
        self.skeleton_function = skeleton_function
        self.decay_function = decay_function
        self.neighborhood = neighborhood
        self.agent_id = agent_id
        self._grid = np.zeros((self.d_width, self.d_height), dtype=np.int16)
        self._update_grid()
        self.ros_publisher = ros_publisher

    @classmethod
    def from_binary_occ_grid(
        cls,
        binary_occ_grid,
        ros_publisher: t.Optional["rp.RosPublisher"] = None,
        skeleton_function=skeleteton_social_cost_function,
        decay_function=exp_decay_function,
        neighborhood=utils.CHESSBOARD_NEIGHBORHOOD,
        agent_id="",
    ):
        return cls(
            binary_occ_grid.get_grid(),
            binary_occ_grid.d_width,
            binary_occ_grid.d_height,
            binary_occ_grid.cell_size,
            binary_occ_grid.grid_pose,
            binary_occ_grid.inflation_radius,
            ros_publisher=ros_publisher,
            skeleton_function=skeleton_function,
            decay_function=decay_function,
            neighborhood=neighborhood,
            agent_id="",
        )

    def _update_grid(self):
        self._grid = compute_social_costmap(
            binary_occ_grid=self.occupation_grid,
            cell_size=self.cell_size,
            ros_publisher=self.ros_publisher,
            agent_id=self.agent_id,
        )

    def get_grid(self):
        return self._grid
