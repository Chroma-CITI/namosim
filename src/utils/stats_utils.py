from src.behaviors.plan.action_result import ActionSuccess
from src.utils.utils import euclidean_distance
import src.utils.connectivity as connectivity
from src.worldreps.occupation_based.binary_occupancy_grid import BinaryOccupancyGrid, BinaryInflatedOccupancyGrid
from src.worldreps.occupation_based.social_topological_occupation_cost_grid import compute_social_costmap
from src.display.ros_publisher import RosPublisher
from src.utils import utils


def get_reallocated_obstacles(init_world, end_world):
    reallocated_obstacles_uids = []
    end_entities = end_world.entities
    for init_entity_uid, init_entity in init_world.entities.items():
        if init_entity.pose != end_entities[init_entity_uid].pose:
            reallocated_obstacles_uids.append(init_entity_uid)
    return reallocated_obstacles_uids


def get_nb_reallocated_obstacles(init_world, end_world):
    return len(get_reallocated_obstacles(init_world, end_world))


def get_transferred_obstacles_set(actions_results):
    transferred_obstacles = set()
    for action_result in actions_results:
        if isinstance(action_result, ActionSuccess) and action_result.is_transfer:
            transferred_obstacles.add(action_result.obstacle_uid)
    return transferred_obstacles


def get_transferred_obstacles_sequence(actions_results):
    transferred_obstacles = []
    for action_result in actions_results:
        action = action_result.action
        if isinstance(action_result, ActionSuccess) and action_result.is_transfer:
            if len(transferred_obstacles) >= 1:
                if action_result.obstacle_uid != transferred_obstacles[-1]:
                    transferred_obstacles.append(action_result.obstacle_uid)
            else:
                transferred_obstacles.append(action.obstacle_uid)
    return transferred_obstacles


def get_nb_transferred_obstacles(actions_results):
    return len(get_transferred_obstacles_set(actions_results))


def get_total_path_lengths(actions_results):
    transit_path_length = 0.
    transfer_path_length = 0.

    if len(actions_results) >= 2:
        action_result_iter = iter(actions_results)
        prev_action_result = next(action_result_iter)
        for action_result in action_result_iter:
            if isinstance(action_result, ActionSuccess):
                cur_pose = action_result.robot_pose
                prev_pose = prev_action_result.robot_pose
                if action_result.is_transfer:
                    transfer_path_length += euclidean_distance(cur_pose, prev_pose)
                else:
                    transit_path_length += euclidean_distance(cur_pose, prev_pose)
                prev_action_result = action_result

    return transit_path_length, transfer_path_length


def get_total_transit_path_length(actions_results):
    return get_total_path_lengths(actions_results)[0]


def get_total_transfer_path_length(actions_results):
    return get_total_path_lengths(actions_results)[1]


def get_transit_transfer_ratio(actions_results):
    transit_path_length, transfer_path_length = get_total_path_lengths(actions_results)
    try:
        return transit_path_length / transfer_path_length
    except ZeroDivisionError:
        return float("inf")


def get_connectivity_stats(world, inflation_radius, entities_to_ignore):
    polygons = {uid: e.polygon for uid, e in world.entities.items() if uid not in entities_to_ignore}
    occ_grid = BinaryInflatedOccupancyGrid(
        polygons, world.dd.res, inflation_radius, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD
    )
    ccs_data = connectivity.CCSData(
        *connectivity.init_ccs_for_grid(occ_grid.grid, occ_grid.d_width, occ_grid.d_height, occ_grid.neighborhood)
    )
    connected_components = ccs_data.ccs
    connected_components_grid = ccs_data.ccs_grid
    RosPublisher().publish_connected_components_grid(connected_components_grid, world.dd, ns='simulation')

    # cc is abbreviation of connected component
    nb_cc = len(connected_components)

    biggest_cc_size, all_cc_sum_size = 0, 0
    for cc in connected_components.values():
        all_cc_sum_size += len(cc.visited)
        if len(cc.visited) > biggest_cc_size:
            biggest_cc_size = len(cc.visited)

    frag_percentage = 0 if all_cc_sum_size == 0 else (1. - float(biggest_cc_size) / float(all_cc_sum_size)) * 100.

    return nb_cc, biggest_cc_size, all_cc_sum_size, frag_percentage


def get_social_costs_stats(world, entities_to_compute_social_cost_for, ):
    polygons = {uid: e.polygon for uid, e in world.entities.items() if uid not in entities_to_compute_social_cost_for}
    occ_grid = BinaryOccupancyGrid(polygons, world.dd.res, neighborhood=utils.CHESSBOARD_NEIGHBORHOOD)
    abs_social_costmap = compute_social_costmap(occ_grid.grid, occ_grid.res, log_costmaps=False, ns='simulation')

    absolute_social_cost = 0.
    for entity_uid in entities_to_compute_social_cost_for:
        entity = world.entities[entity_uid]
        entity_cell_set = utils.polygon_to_discrete_cells_set(
            entity.polygon, occ_grid.res, occ_grid.grid_pose, occ_grid.d_width, occ_grid.d_height, occ_grid.r_width, occ_grid.r_height
        )
        for cell in entity_cell_set:
            absolute_social_cost += abs_social_costmap[cell[0]][cell[1]]

    return absolute_social_cost


def relative_change(init_value, end_value, return_percentage=True):
    if init_value == end_value:
        return 0.
    else:
        if init_value == 0.:
            return 1.
        else:
            return (float(end_value) / float(init_value) - 1.) * (100. if return_percentage else 1.)
