import copy
import typing as t

import namosim.navigation.action_result as ar
import namosim.utils.connectivity as connectivity
from namosim.display.ros2_publisher import RosPublisher
from namosim.utils.utils import euclidean_distance
from namosim.world.social_topological_occupation_cost_grid import (
    compute_social_costmap,
)
from namosim.world.world import World


def get_reallocated_obstacles(init_world: World, end_world: World):
    reallocated_obstacles_uids: t.List[str] = []
    end_entities = end_world.dynamic_entities
    for init_entity_uid, init_entity in init_world.dynamic_entities.items():
        if init_entity.pose != end_entities[init_entity_uid].pose:
            reallocated_obstacles_uids.append(init_entity_uid)
    return reallocated_obstacles_uids


def get_nb_reallocated_obstacles(init_world: World, end_world: World):
    return len(get_reallocated_obstacles(init_world, end_world))


def get_transferred_obstacles_set(actions_results: t.List[ar.ActionResult]):
    transferred_obstacles: t.Set[str] = set()
    for action_result in actions_results:
        if (
            isinstance(action_result, ar.ActionSuccess)
            and action_result.is_transfer
            and action_result.obstacle_uid is not None
        ):
            transferred_obstacles.add(action_result.obstacle_uid)
    return transferred_obstacles


def get_transferred_obstacles_sequence(actions_results: t.List[ar.ActionResult]):
    transferred_obstacles: t.List[str] = []
    for action_result in actions_results:
        if (
            isinstance(action_result, ar.ActionSuccess)
            and action_result.is_transfer
            and action_result.obstacle_uid is not None
        ):
            if len(transferred_obstacles) >= 1:
                if action_result.obstacle_uid != transferred_obstacles[-1]:
                    transferred_obstacles.append(action_result.obstacle_uid)
            else:
                transferred_obstacles.append(action_result.obstacle_uid)
    return transferred_obstacles


def get_nb_transferred_obstacles(actions_results: t.List[ar.ActionResult]):
    return len(get_transferred_obstacles_set(actions_results))


def get_total_path_lengths(actions_results: t.List[ar.ActionResult]):
    transit_path_length = 0.0
    transfer_path_length = 0.0

    if len(actions_results) >= 2:
        action_result_iter = iter(actions_results)
        prev_action_result = next(action_result_iter)
        for action_result in action_result_iter:
            if isinstance(action_result, ar.ActionSuccess) and isinstance(
                prev_action_result, ar.ActionSuccess
            ):
                cur_pose = action_result.robot_pose
                prev_pose = prev_action_result.robot_pose
                prev_action_result = action_result

                if cur_pose is None or prev_pose is None:
                    continue

                if action_result.is_transfer:
                    transfer_path_length += euclidean_distance(cur_pose, prev_pose)
                else:
                    transit_path_length += euclidean_distance(cur_pose, prev_pose)

    return transit_path_length, transfer_path_length


def get_total_transit_path_length(actions_results: t.List[ar.ActionResult]):
    return get_total_path_lengths(actions_results)[0]


def get_total_transfer_path_length(actions_results: t.List[ar.ActionResult]):
    return get_total_path_lengths(actions_results)[1]


def get_transit_transfer_ratio(actions_results: t.List[ar.ActionResult]):
    transit_path_length, transfer_path_length = get_total_path_lengths(actions_results)
    try:
        return transit_path_length / transfer_path_length
    except ZeroDivisionError:
        return float("inf")


def get_connectivity_stats(
    world: World,
    inflation_radius: float,
    entities_to_ignore: t.Set[str],
):
    occ_grid = world.get_dynamic_occupancy_grid(
        inflation_radius=inflation_radius, ignored_entities=entities_to_ignore
    )
    ccs_data = connectivity.init_ccs_for_grid(
        occ_grid.grid, occ_grid.d_width, occ_grid.d_height, occ_grid.neighborhood
    )
    connected_components = ccs_data.ccs

    # cc is abbreviation of connected component
    nb_cc = len(connected_components)

    biggest_cc_size, all_cc_sum_size = 0, 0
    for cc in connected_components.values():
        all_cc_sum_size += len(cc.visited)
        if len(cc.visited) > biggest_cc_size:
            biggest_cc_size = len(cc.visited)

    frag_percentage = (
        0
        if all_cc_sum_size == 0
        else (1.0 - float(biggest_cc_size) / float(all_cc_sum_size)) * 100.0
    )

    return nb_cc, biggest_cc_size, all_cc_sum_size, frag_percentage


def get_social_costs_stats(
    world: World,
    entities_to_compute_social_cost_for: t.Set[str],
):
    polygons = {
        uid: e.polygon
        for uid, e in world.dynamic_entities.items()
        if uid not in entities_to_compute_social_cost_for
    }
    occ_grid = copy.deepcopy(world.map)
    occ_grid.update_polygons(polygons)

    abs_social_costmap = compute_social_costmap(
        binary_occ_grid=occ_grid.grid,
        cell_size=occ_grid.cell_size,
        log_costmaps=False,
        agent_id="simulation",
    )

    absolute_social_cost = 0.0
    for entity_uid in entities_to_compute_social_cost_for:
        entity = world.dynamic_entities[entity_uid]
        entity_cell_set = occ_grid.rasterize_polygon(
            entity.polygon,
        )
        for cell in entity_cell_set:
            absolute_social_cost += abs_social_costmap[cell[0]][cell[1]]

    return float(absolute_social_cost)


def relative_change(
    init_value: float, end_value: float, return_percentage: bool = True
):
    if init_value == end_value:
        return 0.0
    if init_value == 0.0:
        return 1.0
    return (float(end_value) / float(init_value) - 1.0) * (
        100.0 if return_percentage else 1.0
    )
