import json
import plotly.graph_objects as go
import plotly.subplots as sp
import os
import numpy as np


def aggregate_goals(scenario_report_data):
    # Initialize criteria
    agg_data = {
        "absolute_social_cost": [scenario_report_data['absolute_social_cost_initial']],
        "biggest_free_component_size": [scenario_report_data['biggest_free_component_size_initial']],
        "free_space_size": [scenario_report_data['free_space_size_initial']],
        "number_of_connected_components": [scenario_report_data['number_of_connected_components_initial']],
        "space_fragmentation_percentage": [scenario_report_data['space_fragmentation_percentage_initial']],
        "transit_path_length": [0.],
        "transfer_path_length": [0.],
        "total_path_length": [0.],
        "number_of_transferred_obstacles": [0],
        "number_of_failed_goals": [0],
        "cumulated_transit_path_length": [0.],
        "cumulated_transfer_path_length": [0.],
        "cumulated_total_path_length": [0.],
        "cumulated_number_of_transferred_obstacles": [0],
        "cumulated_number_of_failed_goals": [0]
    }

    # Add each goal's criteria values to the aggregation variable
    goals_reports = scenario_report_data['agents'][0]['goals_reports']
    for goal_report in goals_reports:
        goal_fail = 1 if goal_report['goal_status'] == 'failure' else 0

        agg_data["absolute_social_cost"].append(goal_report['absolute_social_cost_after_goal'])
        agg_data["biggest_free_component_size"].append(goal_report['biggest_free_component_size_after_goal'])
        agg_data["free_space_size"].append(goal_report['free_space_size_after_goal'])
        agg_data["number_of_connected_components"].append(goal_report['number_of_connected_components_after_goal'])
        agg_data["space_fragmentation_percentage"].append(goal_report['space_fragmentation_percentage_after_goal'])

        agg_data["transit_path_length"].append(goal_report['transit_path_length'])
        agg_data["transfer_path_length"].append(goal_report['transfer_path_length'])
        agg_data["total_path_length"].append(goal_report['total_path_length'])
        agg_data["number_of_transferred_obstacles"].append(goal_report['number_of_transferred_obstacles'])
        agg_data["number_of_failed_goals"].append(goal_fail)

        agg_data["cumulated_transit_path_length"].append(
            goal_report['transit_path_length'] + agg_data["cumulated_transit_path_length"][-1]
        )
        agg_data["cumulated_transfer_path_length"].append(
            goal_report['transfer_path_length'] + agg_data["cumulated_transfer_path_length"][-1]
        )
        agg_data["cumulated_total_path_length"].append(
            goal_report['total_path_length'] + agg_data["cumulated_total_path_length"][-1]
        )
        agg_data["cumulated_number_of_transferred_obstacles"].append(
            goal_report['number_of_transferred_obstacles'] + agg_data["cumulated_number_of_transferred_obstacles"][-1]
        )
        agg_data["cumulated_number_of_failed_goals"].append(
            goal_fail + agg_data["cumulated_number_of_failed_goals"][-1]
        )

    return agg_data


def aggregate_scenarios(nb_goals, namo_logs_folder, snamo_logs_folder, nb_failures_max=float("inf")):
    # Get all scenarios individual recorded data
    namo_scenarios_ids = {
        name for name in os.listdir(namo_logs_folder) if os.path.isdir(os.path.join(namo_logs_folder, name))
    }
    snamo_scenarios_ids = {
        name for name in os.listdir(snamo_logs_folder) if os.path.isdir(os.path.join(snamo_logs_folder, name))
    }

    # Only keep scenarios for which we have both NAMO and SNAMO data
    common_scenarios_ids = namo_scenarios_ids.intersection(snamo_scenarios_ids)

    # For each scenario, aggregate the data of all goals so that cumulated data can be computed
    namo_aggregated_goals_per_scenario_data = {}
    snamo_aggregated_goals_per_scenario_data = {}
    x = range(nb_goals)
    for scenario_id in common_scenarios_ids:
        try:
            namo_path = os.path.join(namo_logs_folder, scenario_id, "sim_results.json")
            snamo_path = os.path.join(snamo_logs_folder, scenario_id, "sim_results.json")

            with open(namo_path, "r") as namo_f:
                namo_data = json.load(namo_f)
            with open(snamo_path, "r") as snamo_f:
                snamo_data = json.load(snamo_f)

            if "Exceptions" in namo_data or "Exceptions" in snamo_data:
                continue

            nb_failure_namo = len(
                [True for gr in namo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
            if nb_failure_namo > nb_failures_max:
                continue

            nb_failure_snamo = len(
                [True for gr in snamo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
            if nb_failure_snamo > nb_failures_max:
                continue

        except IOError as e:
            continue

        namo_aggregated_goals_per_scenario_data[scenario_id] = aggregate_goals(namo_data)
        snamo_aggregated_goals_per_scenario_data[scenario_id] = aggregate_goals(snamo_data)

    # Finally, we aggregate once more the aggregated data, but this time, for all scenarios
    criteria_ids = next(iter(namo_aggregated_goals_per_scenario_data.values())).keys()
    namo_aggregated_scenarios_data = {criterion_id: [[] for i in x] for criterion_id in criteria_ids}
    snamo_aggregated_scenarios_data = {criterion_id: [[] for i in x] for criterion_id in criteria_ids}
    index_to_scenario_id = []
    for scenario_id, namo_scenario_data in namo_aggregated_goals_per_scenario_data.items():
        snamo_scenario_data = snamo_aggregated_goals_per_scenario_data[scenario_id]
        index_to_scenario_id.append(scenario_id)
        for i in x:
            for criterion_id in namo_scenario_data.keys():
                namo_aggregated_scenarios_data[criterion_id][i].append(namo_scenario_data[criterion_id][i])

            for criterion_id in namo_scenario_data.keys():
                snamo_aggregated_scenarios_data[criterion_id][i].append(snamo_scenario_data[criterion_id][i])

    return {
        "namo_aggregated_goals_per_scenario_data": namo_aggregated_goals_per_scenario_data,
        "snamo_aggregated_goals_per_scenario_data": snamo_aggregated_goals_per_scenario_data,
        "namo_aggregated_scenarios_data": namo_aggregated_scenarios_data,
        "snamo_aggregated_scenarios_data": snamo_aggregated_scenarios_data,
        "index_to_scenario_id": index_to_scenario_id
    }


def plot_criterion(namo_data, snamo_data, criterion_id, criterion_name, nb_goals,
                   save_filepath=None, show=True, padding_percentage=0.05):

    # Compute x-axis values
    x = range(nb_goals)

    # Compute min and max y values
    min_y = min([min(namo_data[criterion_id][i]) for i in x] + [min(snamo_data[criterion_id][i]) for i in x])
    max_y = max([max(namo_data[criterion_id][i]) for i in x] + [max(snamo_data[criterion_id][i]) for i in x])

    # Compute distribution boxes
    absolute_social_cost_boxes_namo = [
        go.Box(y=namo_data[criterion_id][i], name=str(i), marker_color="blue", boxmean='sd', boxpoints=False)
        for i in x
    ]
    absolute_social_cost_boxes_snamo = [
        go.Box(y=snamo_data[criterion_id][i], name=str(i), marker_color="green", boxmean='sd', boxpoints=False)
        for i in x
    ]

    # Compute median scatter plots
    absolute_social_cost_median_namo = [np.median(namo_data[criterion_id][i]) for i in x]
    absolute_social_cost_median_scatter_namo = go.Scatter(
        x=x, y=absolute_social_cost_median_namo, mode='lines+markers',
        name=criterion_name + ' - Median of all scenarios - NAMO', marker_color="blue"
    )
    absolute_social_cost_median_snamo = [np.median(snamo_data[criterion_id][i]) for i in x]
    absolute_social_cost_median_scatter_snamo = go.Scatter(
        x=x, y=absolute_social_cost_median_snamo, mode='lines+markers',
        name=criterion_name + ' - Median of all scenarios - SNAMO', marker_color="green"
    )

    # Compute mean scatter plots
    absolute_social_cost_mean_namo = [np.mean(namo_data[criterion_id][i]) for i in x]
    absolute_social_cost_mean_scatter_namo = go.Scatter(
        x=x, y=absolute_social_cost_mean_namo, mode='lines+markers',
        name=criterion_name + ' - Mean of all scenarios - NAMO', marker_color="blue"
    )
    absolute_social_cost_mean_snamo = [np.median(snamo_data[criterion_id][i]) for i in x]
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
        fig_social_cost.write_html(save_filepath)


def plot_data(aggregated_data, nb_goals, saved_plots_path, show=True, padding_percentage=0.05):
    namo_data = aggregated_data["namo_aggregated_scenarios_data"]
    snamo_data = aggregated_data["snamo_aggregated_scenarios_data"]

    criterion_id_to_criterion_name = {
        "absolute_social_cost": "Absolute Social Cost (Arbitrary Units)",
        "biggest_free_component_size": "Biggest Free Space Component Size (Number of cells)",
        "free_space_size": "Total Free Space Size (Number of cells)",
        "number_of_connected_components": "Number of Connected Components",
        "space_fragmentation_percentage": "Space fragmentation percentage",
        "transit_path_length": "Transit Paths Length (meters)",
        "transfer_path_length": "Transfer Paths Length (meters)",
        "total_path_length": "Total Paths Length (meters)",
        "number_of_transferred_obstacles": "Number of Obstacles Transfers",
        "number_of_failed_goals": "Number of Failed Goals",
        "cumulated_transit_path_length": "Cumulated Transit Paths Length (meters)",
        "cumulated_transfer_path_length": "Cumulated Transfer Paths Length (meters)",
        "cumulated_total_path_length": "Cumulated Total Paths Length (meters)",
        "cumulated_number_of_transferred_obstacles": "Cumulated Number of Obstacles Transfers",
        "cumulated_number_of_failed_goals": "Cumulated Number of Failed Goals"
    }

    for criterion_id, criterion_name in criterion_id_to_criterion_name.items():
        plot_criterion(
            namo_data, snamo_data, criterion_id, criterion_name,
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
        aggregated_data = aggregate_scenarios(nb_goals, namo_logs_folder, snamo_logs_folder, nb_failures_max=250)
        with open(os.path.join(main_dirname, "synthesis.json"), "w") as f:
            json.dump(aggregated_data, f)

    plot_data(aggregated_data, nb_goals, main_dirname, show=True, padding_percentage=0.05)
