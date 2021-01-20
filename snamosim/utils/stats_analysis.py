import json
import plotly.graph_objects as go
import plotly.subplots as sp
import os
import copy
import numpy as np

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

    space_fragmentation_percentage = [data['space_fragmentation_percentage_initial']] + [
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


def compute_aggregated_data(nb_goals, namo_logs_folder, snamo_logs_folder):
    namo_dirnames = {name for name in os.listdir(namo_logs_folder) if
                     os.path.isdir(os.path.join(namo_logs_folder, name))}
    snamo_dirnames = {name for name in os.listdir(snamo_logs_folder) if
                      os.path.isdir(os.path.join(snamo_logs_folder, name))}

    common_dirnames = namo_dirnames.intersection(snamo_dirnames)

    namo_aggregated_data = {}
    snamo_aggregated_data = {}

    x = range(nb_goals)

    for dirname in common_dirnames:
        try:
            namo_path = os.path.join(namo_logs_folder, dirname, "sim_results.json")
            snamo_path = os.path.join(snamo_logs_folder, dirname, "sim_results.json")

            with open(namo_path, "r") as namo_f:
                namo_data = json.load(namo_f)
            with open(snamo_path, "r") as snamo_f:
                snamo_data = json.load(snamo_f)

            if "Exceptions" in namo_data or "Exceptions" in snamo_data:
                continue

            nb_failure_namo = len(
                [True for gr in namo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
            if nb_failure_namo > 50:
                continue

            nb_failure_snamo = len(
                [True for gr in snamo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
            if nb_failure_snamo > 50:
                continue

        except IOError as e:
            continue

        namo_aggregated_data[dirname] = compute_graph_data(namo_data)
        snamo_aggregated_data[dirname] = compute_graph_data(snamo_data)

    namo_final_data = {
        "absolute_social_cost": [[] for i in x],
        "biggest_free_component_size": [[] for i in x],
        "free_space_size": [[] for i in x],
        "number_of_connected_components": [[] for i in x],
        "space_fragmentation_percentage": [[] for i in x],
        "cumulated_transit_path_length": [[] for i in x],
        "cumulated_transfer_path_length": [[] for i in x],
        "cumulated_total_path_length": [[] for i in x],
        "cumulated_number_of_transferred_obstacles": [[] for i in x]
    }

    snamo_final_data = copy.deepcopy(namo_final_data)
    index_to_dirname = []

    for dirname, namo_graph_data in namo_aggregated_data.items():
        snamo_graph_data = snamo_aggregated_data[dirname]

        index_to_dirname.append(dirname)

        for i in x:
            namo_final_data["absolute_social_cost"][i].append(
                namo_graph_data["absolute_social_cost"][i]),
            namo_final_data["biggest_free_component_size"][i].append(
                namo_graph_data["biggest_free_component_size"][i]),
            namo_final_data["free_space_size"][i].append(
                namo_graph_data["absolute_social_cost"][i]),
            namo_final_data["space_fragmentation_percentage"][i].append(
                namo_graph_data["space_fragmentation_percentage"][i]),
            namo_final_data["cumulated_transit_path_length"][i].append(
                namo_graph_data["cumulated_transit_path_length"][i]),
            namo_final_data["cumulated_transfer_path_length"][i].append(
                namo_graph_data["cumulated_transfer_path_length"][i]),
            namo_final_data["cumulated_total_path_length"][i].append(
                namo_graph_data["cumulated_total_path_length"][i]),
            namo_final_data["cumulated_number_of_transferred_obstacles"][i].append(
                namo_graph_data["cumulated_number_of_transferred_obstacles"][i])

            snamo_final_data["absolute_social_cost"][i].append(
                snamo_graph_data["absolute_social_cost"][i]),
            snamo_final_data["biggest_free_component_size"][i].append(
                snamo_graph_data["biggest_free_component_size"][i]),
            snamo_final_data["free_space_size"][i].append(
                snamo_graph_data["absolute_social_cost"][i]),
            snamo_final_data["number_of_connected_components"][i].append(
                snamo_graph_data["number_of_connected_components"][i]),
            snamo_final_data["space_fragmentation_percentage"][i].append(
                snamo_graph_data["space_fragmentation_percentage"][i]),
            snamo_final_data["cumulated_transit_path_length"][i].append(
                snamo_graph_data["cumulated_transit_path_length"][i]),
            snamo_final_data["cumulated_transfer_path_length"][i].append(
                snamo_graph_data["cumulated_transfer_path_length"][i]),
            snamo_final_data["cumulated_total_path_length"][i].append(
                snamo_graph_data["cumulated_total_path_length"][i]),
            snamo_final_data["cumulated_number_of_transferred_obstacles"][i].append(
                snamo_graph_data["cumulated_number_of_transferred_obstacles"][i])

    aggregated_data = {
        "namo_aggregated_data": namo_aggregated_data,
        "snamo_aggregated_data": snamo_aggregated_data,
        "namo_final_data": namo_final_data,
        "snamo_final_data": snamo_final_data,
        "index_to_dirname": index_to_dirname
    }

    return aggregated_data


def plot_criterion(namo_final_data, snamo_final_data, criterion_id, criterion_name, nb_goals,
                   save_filepath=None, show=True, padding_percentage=0.05):

    # Compute x-axis values
    x = range(nb_goals)

    # Compute min and max y values
    min_y = min(
        [min(namo_final_data[criterion_id][i]) for i in x]
        + [min(snamo_final_data[criterion_id][i]) for i in x]
    )
    max_y = max(
        [max(namo_final_data[criterion_id][i]) for i in x]
        + [max(snamo_final_data[criterion_id][i]) for i in x]
    )

    # Compute distribution boxes
    absolute_social_cost_boxes_namo = [
        go.Box(y=namo_final_data[criterion_id][i], name=str(i), marker_color="blue", boxmean='sd', boxpoints=False)
        for i in x
    ]
    absolute_social_cost_boxes_snamo = [
        go.Box(y=snamo_final_data[criterion_id][i], name=str(i), marker_color="green", boxmean='sd', boxpoints=False)
        for i in x
    ]

    # Compute median scatter plots
    absolute_social_cost_median_namo = [np.median(namo_final_data[criterion_id][i]) for i in x]
    absolute_social_cost_median_scatter_namo = go.Scatter(
        x=x, y=absolute_social_cost_median_namo, mode='lines+markers',
        name=criterion_name + ' - Median of all scenarios - NAMO', marker_color="blue"
    )
    absolute_social_cost_median_snamo = [np.median(snamo_final_data[criterion_id][i]) for i in x]
    absolute_social_cost_median_scatter_snamo = go.Scatter(
        x=x, y=absolute_social_cost_median_snamo, mode='lines+markers',
        name=criterion_name + ' - Median of all scenarios - SNAMO', marker_color="green"
    )

    # Compute mean scatter plots
    absolute_social_cost_mean_namo = [np.mean(namo_final_data[criterion_id][i]) for i in x]
    absolute_social_cost_mean_scatter_namo = go.Scatter(
        x=x, y=absolute_social_cost_mean_namo, mode='lines+markers',
        name=criterion_name + ' - Mean of all scenarios - NAMO', marker_color="blue"
    )
    absolute_social_cost_mean_snamo = [np.median(snamo_final_data[criterion_id][i]) for i in x]
    absolute_social_cost_mean_scatter_snamo = go.Scatter(
        x=x, y=absolute_social_cost_mean_snamo, mode='lines+markers',
        name=criterion_name + ' - Mean of all scenarios - SNAMO', marker_color="green"
    )

    # Create figure with 4 subplots in a grid
    fig_social_cost = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Distribution accross draws for NAMO", "Distribution accross draws for S-NAMO",
            "Side-by-side medians for NAMO (blue) and S-NAMO (green)",
            "Side-by-side means for NAMO (blue) and S-NAMO (green)"
        )
    )

    # Add distribution of values for NAMO (top-left subfig)
    for asc_box_namo in absolute_social_cost_boxes_namo:
        fig_social_cost.add_trace(asc_box_namo, 1, 1)

    # Add distribution of values for SNAMO (top-right subfig)
    for asc_box_snamo in absolute_social_cost_boxes_snamo:
        fig_social_cost.add_trace(asc_box_snamo, 1, 2)

    # Add median scatter plot (bottom-left subfig)
    fig_social_cost.add_trace(absolute_social_cost_median_scatter_namo, 2, 1)
    fig_social_cost.add_trace(absolute_social_cost_median_scatter_snamo, 2, 1)

    # Add median scatter plot (bottom-right subfig)
    fig_social_cost.add_trace(absolute_social_cost_mean_scatter_namo, 2, 2)
    fig_social_cost.add_trace(absolute_social_cost_mean_scatter_snamo, 2, 2)

    # Update xaxis properties
    fig_social_cost.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=1, col=1)
    fig_social_cost.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=1, col=2)
    fig_social_cost.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=2, col=1)
    fig_social_cost.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=2, col=2)

    # Update yaxis properties
    min_mult, max_mult = 1. - padding_percentage, 1. + padding_percentage
    fig_social_cost.update_yaxes(title_text=criterion_name, range=[min_y * min_mult, max_y * max_mult], row=1, col=1)
    fig_social_cost.update_yaxes(title_text=criterion_name, range=[min_y * min_mult, max_y * max_mult], row=1, col=2)
    fig_social_cost.update_yaxes(title_text=criterion_name, range=[min_y * min_mult, max_y * max_mult], row=2, col=1)
    fig_social_cost.update_yaxes(title_text=criterion_name, range=[min_y * min_mult, max_y * max_mult], row=2, col=2)

    # Set title and display
    fig_social_cost.update_layout(
        showlegend=False,
        title_text=criterion_name + " accross multiple draws of 200 random goals",
        title_x=0.5
    )

    if show:
        fig_social_cost.show()
    if save_filepath:
        # fig_social_cost.write_html("/home/xia0ben/INRIA/Code/s-namo-sim/logs/04_after_the_feast/absolute_social_cost.html")
        fig_social_cost.write_html(save_filepath)


def plot_data(aggregated_data, nb_goals, saved_plots_path, show=True, padding_percentage=0.05):
    namo_final_data, snamo_final_data = aggregated_data["namo_final_data"], aggregated_data["snamo_final_data"]

    criterion_id_to_criterion_name = {
        "absolute_social_cost": "Absolute Social Cost (Arbitrary Units)",
        "biggest_free_component_size": "Biggest Free Space Component Size (Number of cells)",
        "free_space_size": "Total Free Space Size (Number of cells)",
        "number_of_connected_components": "Number of Connected Components",
        "space_fragmentation_percentage": "Space fragmentation percentage",
        "cumulated_transit_path_length": "Cumulated Transit Paths Length (meters)",
        "cumulated_transfer_path_length": "Cumulated Transfer Paths Length (meters)",
        "cumulated_total_path_length": "Cumulated Total Paths Length (meters)",
        "cumulated_number_of_transferred_obstacles": "Cumulated Number of Transferred Obstacles"
    }

    for criterion_id, criterion_name in criterion_id_to_criterion_name.items():
        plot_criterion(
            namo_final_data, snamo_final_data, criterion_id, criterion_name,
            nb_goals, save_filepath=os.path.join(saved_plots_path, criterion_id+".html"),
            show=show, padding_percentage=padding_percentage
        )


if __name__ == "__main__":

    nb_goals = 200
    main_dirname = os.path.join(os.path.dirname(__file__), "../../logs/04_after_the_feast/")

    try:
        with open(os.path.join(main_dirname, "synthesis.json"), "r") as f:
            aggregated_data = json.load(f)
    except IOError as e:

        namo_logs_folder = os.path.join(main_dirname, "stilman_2005_behavior_complexified_random_goal_no_reset/")
        snamo_logs_folder = os.path.join(main_dirname, "stilman_2005_behavior_complexified_random_goal_no_reset_snamo/")

        aggregated_data = compute_aggregated_data(nb_goals, namo_logs_folder, snamo_logs_folder)
        with open(os.path.join(main_dirname, "synthesis.json"), "w") as f:
            json.dump(aggregated_data, f)

    plot_data(aggregated_data, nb_goals, main_dirname)
