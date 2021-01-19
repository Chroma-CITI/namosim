import json
import plotly.graph_objects as go
import numpy as np
import os
import copy
import plotly.express as px


def compute_graph_data(data):

    goals_reports = data['agents'][0]['goals_reports']

    absolute_social_cost = [data['absolute_social_cost_initial']] + [
        goal_report['absolute_social_cost_after_goal']
        for goal_report in goals_reports
    ]

    biggest_free_component_size = [data['biggest_free_component_size_initial']] + [
        goal_report['biggest_free_component_size_after_goal']
        for goal_report in goals_reports
    ]

    free_space_size = [data['free_space_size_initial']] + [
        goal_report['free_space_size_after_goal']
        for goal_report in goals_reports
    ]

    number_of_connected_components = [data['number_of_connected_components_initial']] + [
        goal_report['number_of_connected_components_after_goal']
        for goal_report in goals_reports
    ]

    space_fragmentation_percentage =  [data['space_fragmentation_percentage_initial']] + [
        goal_report['space_fragmentation_percentage_after_goal']
        for goal_report in goals_reports
    ]

    cumulated_transit_path_length = [0.]
    cumulated_transfer_path_length = [0.]
    cumulated_total_path_length = [0.]
    cumulated_number_of_transferred_obstacles = [0]
    for goal_report in goals_reports:
        cumulated_transit_path_length.append(
            goal_report['transit_path_length'] + cumulated_transit_path_length[-1]
        )
        cumulated_transfer_path_length.append(
            goal_report['transfer_path_length'] + cumulated_transfer_path_length[-1]
        )
        cumulated_total_path_length.append(
            goal_report['total_path_length'] + cumulated_total_path_length[-1]
        )
        cumulated_number_of_transferred_obstacles.append(
            goal_report['number_of_transferred_obstacles'] + cumulated_number_of_transferred_obstacles[-1]
        )

    return {
        "absolute_social_cost": absolute_social_cost,
        "biggest_free_component_size": biggest_free_component_size,
        "free_space_size": free_space_size,
        "number_of_connected_components": number_of_connected_components,
        "space_fragmentation_percentage" : space_fragmentation_percentage,
        "cumulated_transit_path_length": cumulated_transit_path_length,
        "cumulated_transfer_path_length": cumulated_transfer_path_length,
        "cumulated_total_path_length": cumulated_total_path_length,
        "cumulated_number_of_transferred_obstacles": cumulated_number_of_transferred_obstacles
    }


if __name__ == "__main__":
    namo_logs_folder = "/home/xia0ben/INRIA/Code/s-namo-sim/logs/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset/"
    snamo_logs_folder = "/home/xia0ben/INRIA/Code/s-namo-sim/logs/04_after_the_feast/stilman_2005_behavior_complexified_random_goal_no_reset_snamo/"

    namo_dirnames = {name for name in os.listdir(namo_logs_folder) if os.path.isdir(os.path.join(namo_logs_folder, name))}
    snamo_dirnames = {name for name in os.listdir(snamo_logs_folder) if
                      os.path.isdir(os.path.join(snamo_logs_folder, name))}

    common_dirnames = namo_dirnames.intersection(snamo_dirnames)

    namo_aggregated_data = {}
    snamo_aggregated_data = {}

    x = range(200)

    for dirname in common_dirnames:
        try:
            namo_path = os.path.join(namo_logs_folder, dirname, "sim_results.json")
            snamo_path = os.path.join(snamo_logs_folder, dirname, "sim_results.json")

            with open(namo_path, "r") as namo_f:
                namo_data = json.load(namo_f)
            with open(snamo_path, "r") as snamo_f:
                snamo_data = json.load(snamo_f)
        except IOError as e:
            continue

        namo_aggregated_data[dirname] = compute_graph_data(namo_data)
        snamo_aggregated_data[dirname] = compute_graph_data(snamo_data)

    namo_final_data = {
        "absolute_social_cost": [[] for i in range(200)],
        "biggest_free_component_size": [[] for i in range(200)],
        "free_space_size": [[] for i in range(200)],
        "number_of_connected_components": [[] for i in range(200)],
        "space_fragmentation_percentage": [[] for i in range(200)],
        "cumulated_transit_path_length": [[] for i in range(200)],
        "cumulated_transfer_path_length": [[] for i in range(200)],
        "cumulated_total_path_length": [[] for i in range(200)],
        "cumulated_number_of_transferred_obstacles": [[] for i in range(200)]
    }

    snamo_final_data = copy.deepcopy(namo_final_data)

    for dirname, namo_graph_data in namo_aggregated_data.items():
        snamo_graph_data = snamo_aggregated_data[dirname]
        for i in range(200):
            namo_final_data["absolute_social_cost"][i].append(namo_graph_data["absolute_social_cost"][i]),
            namo_final_data["biggest_free_component_size"][i].append(namo_graph_data["biggest_free_component_size"][i]),
            namo_final_data["free_space_size"][i].append(namo_graph_data["absolute_social_cost"][i]),
            namo_final_data["number_of_connected_components"][i].append(namo_graph_data["number_of_connected_components"][i]),
            namo_final_data["space_fragmentation_percentage"][i].append(namo_graph_data["space_fragmentation_percentage"][i]),
            namo_final_data["cumulated_transit_path_length"][i].append(namo_graph_data["cumulated_transit_path_length"][i]),
            namo_final_data["cumulated_transfer_path_length"][i].append(namo_graph_data["cumulated_transfer_path_length"][i]),
            namo_final_data["cumulated_total_path_length"][i].append(namo_graph_data["cumulated_total_path_length"][i]),
            namo_final_data["cumulated_number_of_transferred_obstacles"][i].append(namo_graph_data["cumulated_number_of_transferred_obstacles"][i])

            snamo_final_data["absolute_social_cost"][i].append(namo_graph_data["absolute_social_cost"][i]),
            snamo_final_data["biggest_free_component_size"][i].append(namo_graph_data["biggest_free_component_size"][i]),
            snamo_final_data["free_space_size"][i].append(namo_graph_data["absolute_social_cost"][i]),
            snamo_final_data["number_of_connected_components"][i].append(namo_graph_data["number_of_connected_components"][i]),
            snamo_final_data["space_fragmentation_percentage"][i].append(namo_graph_data["space_fragmentation_percentage"][i]),
            snamo_final_data["cumulated_transit_path_length"][i].append(namo_graph_data["cumulated_transit_path_length"][i]),
            snamo_final_data["cumulated_transfer_path_length"][i].append(namo_graph_data["cumulated_transfer_path_length"][i]),
            snamo_final_data["cumulated_total_path_length"][i].append(namo_graph_data["cumulated_total_path_length"][i]),
            snamo_final_data["cumulated_number_of_transferred_obstacles"][i].append(namo_graph_data["cumulated_number_of_transferred_obstacles"][i])

    with open('')

    fig_social_cost = go.Figure()
    fig_social_cost.add_trace(go.Box(
        x=x, y=namo_final_data["absolute_social_cost"], name='Absolute social cost (UA))'
    ))
    # fig_social_cost.add_trace(go.Scatter(
    #     x=x, y=absolute_social_cost, mode='lines+markers', name='Absolute social cost (UA))'
    # ))
    fig_social_cost.update_layout(showlegend=True)

    # fig_space_size = go.Figure()
    # fig_space_size.add_trace(go.Box(
    #     x=x, y=namo_final_data["biggest_free_component_size"], name='Absolute social cost (UA))'
    # ))
    # fig_space_size.add_trace(go.Box(
    #     x=x, y=namo_final_data["free_space_size"], name='Absolute social cost (UA))'
    # ))
    #
    # fig_number_of_connected_components = go.Figure()
    # fig_number_of_connected_components.add_trace(go.Box(
    #     x=x, y=namo_final_data["number_of_connected_components"], name='Absolute social cost (UA))'
    # ))
    # fig_number_of_connected_components.update_layout(showlegend=True)
    #
    # fig_space_fragmentation_percentage = go.Figure()
    # fig_space_fragmentation_percentage.add_trace(go.Box(
    #     x=x, y=namo_final_data["space_fragmentation_percentage"], name='Absolute social cost (UA))'
    # ))
    # fig_space_fragmentation_percentage.update_layout(showlegend=True)
    #
    # fig_path_length = go.Figure()
    # fig_path_length.add_trace(go.Box(
    #     x=x, y=namo_final_data["cumulated_transit_path_length"], name='Absolute social cost (UA))'
    # ))
    # fig_path_length.add_trace(go.Box(
    #     x=x, y=namo_final_data["cumulated_transfer_path_length"], name='Absolute social cost (UA))'
    # ))
    # fig_path_length.add_trace(go.Box(
    #     x=x, y=namo_final_data["cumulated_total_path_length"], name='Absolute social cost (UA))'
    # ))
    #
    # fig_nb_transferred_obstacles = go.Figure()
    # fig_nb_transferred_obstacles.add_trace(go.Box(
    #     x=x, y=namo_final_data["cumulated_number_of_transferred_obstacles"], name='Absolute social cost (UA))'
    # ))
    # fig_nb_transferred_obstacles.update_layout(showlegend=True)

    fig_social_cost.show()
    # fig_space_size.show()
    # fig_number_of_connected_components.show()
    # fig_space_fragmentation_percentage.show()
    # fig_path_length.show()
    # fig_nb_transferred_obstacles.show()
