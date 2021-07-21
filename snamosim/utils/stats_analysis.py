import json
import plotly.graph_objects as go
import plotly.subplots as sp
import os
import numpy as np
from snamosim.simulator import AgentStepStats, WorldStepStats, StepStats

# def aggregate_goals(scenario_report_data):
#     # Initialize criteria
#     agg_data = {
#         "absolute_social_cost": [scenario_report_data['absolute_social_cost_initial']],
#         "biggest_free_component_size": [scenario_report_data['biggest_free_component_size_initial']],
#         "free_space_size": [scenario_report_data['free_space_size_initial']],
#         "number_of_connected_components": [scenario_report_data['number_of_connected_components_initial']],
#         "space_fragmentation_percentage": [scenario_report_data['space_fragmentation_percentage_initial']],
#         "transit_path_length": [0.],
#         "transfer_path_length": [0.],
#         "total_path_length": [0.],
#         "number_of_transferred_obstacles": [0],
#         "number_of_failed_goals": [0],
#         "cumulated_transit_path_length": [0.],
#         "cumulated_transfer_path_length": [0.],
#         "cumulated_total_path_length": [0.],
#         "cumulated_number_of_transferred_obstacles": [0],
#         "cumulated_number_of_failed_goals": [0]
#     }
#
#     # Add each goal's criteria values to the aggregation variable
#     goals_reports = scenario_report_data['agents'][0]['goals_reports']
#     for goal_report in goals_reports:
#         goal_fail = 1 if goal_report['goal_status'] == 'failure' else 0
#
#         agg_data["absolute_social_cost"].append(goal_report['absolute_social_cost_after_goal'])
#         agg_data["biggest_free_component_size"].append(goal_report['biggest_free_component_size_after_goal'])
#         agg_data["free_space_size"].append(goal_report['free_space_size_after_goal'])
#         agg_data["number_of_connected_components"].append(goal_report['number_of_connected_components_after_goal'])
#         agg_data["space_fragmentation_percentage"].append(goal_report['space_fragmentation_percentage_after_goal'])
#
#         agg_data["transit_path_length"].append(goal_report['transit_path_length'])
#         agg_data["transfer_path_length"].append(goal_report['transfer_path_length'])
#         agg_data["total_path_length"].append(goal_report['total_path_length'])
#         agg_data["number_of_transferred_obstacles"].append(goal_report['number_of_transferred_obstacles'])
#         agg_data["number_of_failed_goals"].append(goal_fail)
#
#         agg_data["cumulated_transit_path_length"].append(
#             goal_report['transit_path_length'] + agg_data["cumulated_transit_path_length"][-1]
#         )
#         agg_data["cumulated_transfer_path_length"].append(
#             goal_report['transfer_path_length'] + agg_data["cumulated_transfer_path_length"][-1]
#         )
#         agg_data["cumulated_total_path_length"].append(
#             goal_report['total_path_length'] + agg_data["cumulated_total_path_length"][-1]
#         )
#         agg_data["cumulated_number_of_transferred_obstacles"].append(
#             goal_report['number_of_transferred_obstacles'] + agg_data["cumulated_number_of_transferred_obstacles"][-1]
#         )
#         agg_data["cumulated_number_of_failed_goals"].append(
#             goal_fail + agg_data["cumulated_number_of_failed_goals"][-1]
#         )
#
#     return agg_data
#
#
# def aggregate_scenarios(nb_goals, namo_logs_folder, snamo_logs_folder, nb_failures_max=float("inf")):
#     # Get all scenarios individual recorded data
#     namo_scenarios_ids = {
#         name for name in os.listdir(namo_logs_folder) if os.path.isdir(os.path.join(namo_logs_folder, name))
#     }
#     snamo_scenarios_ids = {
#         name for name in os.listdir(snamo_logs_folder) if os.path.isdir(os.path.join(snamo_logs_folder, name))
#     }
#
#     # Only keep scenarios for which we have both NAMO and SNAMO data
#     common_scenarios_ids = namo_scenarios_ids.intersection(snamo_scenarios_ids)
#
#     # For each scenario, aggregate the data of all goals so that cumulated data can be computed
#     namo_aggregated_goals_per_scenario_data = {}
#     snamo_aggregated_goals_per_scenario_data = {}
#     x = range(nb_goals)
#     for scenario_id in common_scenarios_ids:
#         try:
#             namo_path = os.path.join(namo_logs_folder, scenario_id, "sim_results.json")
#
#             with open(namo_path, "r") as namo_f:
#                 namo_data = json.load(namo_f)
#
#             if "Exceptions" in namo_data:
#                 continue
#
#             nb_failure_namo = len(
#                 [True for gr in namo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
#             if nb_failure_namo > nb_failures_max:
#                 continue
#         except Exception as e:
#             if isinstance(e, ValueError):
#                 print("")
#             continue
#
#         try:
#             snamo_path = os.path.join(snamo_logs_folder, scenario_id, "sim_results.json")
#
#             with open(snamo_path, "r") as snamo_f:
#                 snamo_data = json.load(snamo_f)
#
#             if "Exceptions" in snamo_data:
#                 continue
#
#             nb_failure_snamo = len(
#                 [True for gr in snamo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
#             if nb_failure_snamo > nb_failures_max:
#                 continue
#         except Exception as e:
#             if isinstance(e, ValueError):
#                 print("")
#             continue
#
#         namo_aggregated_goals_per_scenario_data[scenario_id] = aggregate_goals(namo_data)
#         snamo_aggregated_goals_per_scenario_data[scenario_id] = aggregate_goals(snamo_data)
#
#     # Finally, we aggregate once more the aggregated data, but this time, for all scenarios
#     criteria_ids = next(iter(namo_aggregated_goals_per_scenario_data.values())).keys()
#     namo_aggregated_scenarios_data = {criterion_id: [[] for i in x] for criterion_id in criteria_ids}
#     snamo_aggregated_scenarios_data = {criterion_id: [[] for i in x] for criterion_id in criteria_ids}
#     index_to_scenario_id = []
#     for scenario_id, namo_scenario_data in namo_aggregated_goals_per_scenario_data.items():
#         snamo_scenario_data = snamo_aggregated_goals_per_scenario_data[scenario_id]
#         index_to_scenario_id.append(scenario_id)
#         for i in x:
#             for criterion_id in namo_scenario_data.keys():
#                 namo_aggregated_scenarios_data[criterion_id][i].append(namo_scenario_data[criterion_id][i])
#
#             for criterion_id in namo_scenario_data.keys():
#                 snamo_aggregated_scenarios_data[criterion_id][i].append(snamo_scenario_data[criterion_id][i])
#
#     return {
#         "namo_aggregated_goals_per_scenario_data": namo_aggregated_goals_per_scenario_data,
#         "snamo_aggregated_goals_per_scenario_data": snamo_aggregated_goals_per_scenario_data,
#         "namo_aggregated_scenarios_data": namo_aggregated_scenarios_data,
#         "snamo_aggregated_scenarios_data": snamo_aggregated_scenarios_data,
#         "index_to_scenario_id": index_to_scenario_id
#     }
#
#
# def plot_criterion(namo_data, snamo_data, criterion_id, criterion_name, nb_goals,
#                    save_filepath=None, show=True, padding_percentage=0.05):
#
#     # Compute x-axis values
#     x = range(nb_goals)
#
#     # Compute min and max y values for distributions
#     min_distribution_y = min([min(namo_data[criterion_id][i]) for i in x] + [min(snamo_data[criterion_id][i]) for i in x])
#     max_distribution_y = max([max(namo_data[criterion_id][i]) for i in x] + [max(snamo_data[criterion_id][i]) for i in x])
#
#     # Compute distribution boxes
#     criterion_boxes_namo = [
#         go.Box(y=namo_data[criterion_id][i], name=str(i), marker_color="blue", boxmean='sd', boxpoints=False)
#         for i in x
#     ]
#     criterion_boxes_snamo = [
#         go.Box(y=snamo_data[criterion_id][i], name=str(i), marker_color="green", boxmean='sd', boxpoints=False)
#         for i in x
#     ]
#
#     # Compute median and mean scatter plots
#     criterion_median_namo = [np.median(namo_data[criterion_id][i]) for i in x]
#     criterion_median_snamo = [np.median(snamo_data[criterion_id][i]) for i in x]
#     criterion_mean_namo = [np.mean(namo_data[criterion_id][i]) for i in x]
#     criterion_mean_snamo = [np.median(snamo_data[criterion_id][i]) for i in x]
#     stats_data = criterion_median_namo + criterion_median_snamo + criterion_mean_namo + criterion_mean_snamo
#     min_mean_median_y = min(stats_data)
#     max_mean_median_y = max(stats_data)
#
#     criterion_median_scatter_namo = go.Scatter(
#         x=x, y=criterion_median_namo, mode='lines+markers',
#         name=criterion_name + ' - Median of all scenarios - NAMO', marker_color="blue"
#     )
#     criterion_median_scatter_snamo = go.Scatter(
#         x=x, y=criterion_median_snamo, mode='lines+markers',
#         name=criterion_name + ' - Median of all scenarios - SNAMO', marker_color="green"
#     )
#     criterion_mean_scatter_namo = go.Scatter(
#         x=x, y=criterion_mean_namo, mode='lines+markers',
#         name=criterion_name + ' - Mean of all scenarios - NAMO', marker_color="blue"
#     )
#     criterion_mean_scatter_snamo = go.Scatter(
#         x=x, y=criterion_mean_snamo, mode='lines+markers',
#         name=criterion_name + ' - Mean of all scenarios - SNAMO', marker_color="green"
#     )
#
#     # Create figure with 4 subplots in a grid
#     fig_criterion = sp.make_subplots(
#         rows=2, cols=2,
#         subplot_titles=(
#             "Distribution accross draws for NAMO", "Distribution accross draws for S-NAMO",
#             "Side-by-side medians for NAMO (blue) and S-NAMO (green)",
#             "Side-by-side means for NAMO (blue) and S-NAMO (green)"
#         )
#     )
#
#     # Add distribution of values for NAMO (top-left subfig)
#     for asc_box_namo in criterion_boxes_namo:
#         fig_criterion.add_trace(asc_box_namo, 1, 1)
#
#     # Add distribution of values for SNAMO (top-right subfig)
#     for asc_box_snamo in criterion_boxes_snamo:
#         fig_criterion.add_trace(asc_box_snamo, 1, 2)
#
#     # Add median scatter plot (bottom-left subfig)
#     fig_criterion.add_trace(criterion_median_scatter_namo, 2, 1)
#     fig_criterion.add_trace(criterion_median_scatter_snamo, 2, 1)
#
#     # Add median scatter plot (bottom-right subfig)
#     fig_criterion.add_trace(criterion_mean_scatter_namo, 2, 2)
#     fig_criterion.add_trace(criterion_mean_scatter_snamo, 2, 2)
#
#     # Update xaxis properties
#     fig_criterion.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=1, col=1)
#     fig_criterion.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=1, col=2)
#     fig_criterion.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=2, col=1)
#     fig_criterion.update_xaxes(title_text="Goal index", range=[0, nb_goals], row=2, col=2)
#
#     # Update yaxis properties
#     min_mult, max_mult = 1. - padding_percentage, 1. + padding_percentage
#     fig_criterion.update_yaxes(title_text=criterion_name, range=[min_distribution_y * min_mult, max_distribution_y * max_mult], row=1, col=1)
#     fig_criterion.update_yaxes(title_text=criterion_name, range=[min_distribution_y * min_mult, max_distribution_y * max_mult], row=1, col=2)
#     fig_criterion.update_yaxes(title_text=criterion_name, range=[min_mean_median_y * min_mult, max_mean_median_y * max_mult], row=2, col=1)
#     fig_criterion.update_yaxes(title_text=criterion_name, range=[min_mean_median_y * min_mult, max_mean_median_y * max_mult], row=2, col=2)
#
#     # Set title and display
#     fig_criterion.update_layout(
#         showlegend=False,
#         title_text=criterion_name + " accross multiple draws of 200 random goals",
#         title_x=0.5
#     )
#
#     if show:
#         fig_criterion.show()
#     if save_filepath:
#         fig_criterion.write_html(save_filepath)
#
#
# def plot_relevant_criteria(namo_data, snamo_data, criterion_id_to_criterion_name, nb_goals,
#                            save_filepath=None, show=True, padding_percentage=0.05):
#     # Plot most relevant criteria into a single plot
#     most_relevant_criteria_to_fig_pose = [
#         "absolute_social_cost",  # (1,1)
#         "biggest_free_component_size",  # (1,2)
#         "free_space_size",
#         "number_of_connected_components",
#         "space_fragmentation_percentage",
#         "cumulated_transit_path_length",
#         "cumulated_total_path_length",
#         "cumulated_number_of_transferred_obstacles"
#         # ,"cumulated_number_of_failed_goals"
#     ]
#     x = range(nb_goals)  # Compute x-axis values
#     rows, cols = 4, 2
#     fig_criteria = sp.make_subplots(
#         rows=rows, cols=cols,
#         subplot_titles=[criterion_id_to_criterion_name[criterion_id] for criterion_id in most_relevant_criteria_to_fig_pose]
#     )
#     for index, criterion_id in enumerate(most_relevant_criteria_to_fig_pose):
#         criterion_name = criterion_id_to_criterion_name[criterion_id]
#
#         criterion_median_namo = [np.median(namo_data[criterion_id][i]) for i in x]
#         criterion_median_snamo = [np.median(snamo_data[criterion_id][i]) for i in x]
#
#         stats_data = criterion_median_namo + criterion_median_snamo
#         min_median_y = min(stats_data)
#         max_median_y = max(stats_data)
#
#         criterion_median_scatter_namo = go.Scatter(x=x, y=criterion_median_namo, mode='lines', marker_color="blue")
#         criterion_median_scatter_snamo = go.Scatter(x=x, y=criterion_median_snamo, mode='lines', marker_color="green")
#
#         row, col = (index // cols + 1, index % cols + 1)
#         fig_criteria.add_trace(criterion_median_scatter_namo, row, col)
#         fig_criteria.add_trace(criterion_median_scatter_snamo, row, col)
#         fig_criteria.update_xaxes(range=[0, nb_goals], row=row, col=col)
#         min_mult, max_mult = 1. - padding_percentage, 1. + padding_percentage
#         fig_criteria.update_yaxes(range=[min_median_y * min_mult, max_median_y * max_mult], row=row, col=col)
#
#     # Set title and display
#     fig_criteria.update_layout(showlegend=False)
#     if show:
#         fig_criteria.show()
#     if save_filepath:
#         fig_criteria.write_html(save_filepath)
#
#
# def plot_data(aggregated_data, saved_plots_path, nb_goals=50, show=True, padding_percentage=0.05, plot_relevant_only=True):
#     namo_data = aggregated_data["namo_aggregated_scenarios_data"]
#     snamo_data = aggregated_data["snamo_aggregated_scenarios_data"]
#
#     criterion_id_to_criterion_name = {
#         "absolute_social_cost": "Absolute Social Cost (Arbitrary Units)",
#         "biggest_free_component_size": "Biggest Free Space Component Size (Number of cells)",
#         "free_space_size": "Total Free Space Size (Number of cells)",
#         "number_of_connected_components": "Number of Connected Components",
#         "space_fragmentation_percentage": "Space fragmentation percentage",
#         "transit_path_length": "Transit Paths Length (meters)",
#         "transfer_path_length": "Transfer Paths Length (meters)",
#         "total_path_length": "Total Paths Length (meters)",
#         "number_of_transferred_obstacles": "Number of Obstacles Transfers",
#         "number_of_failed_goals": "Number of Failed Goals",
#         "cumulated_transit_path_length": "Cumulated Transit Paths Length (meters)",
#         "cumulated_transfer_path_length": "Cumulated Transfer Paths Length (meters)",
#         "cumulated_total_path_length": "Cumulated Total Paths Length (meters)",
#         "cumulated_number_of_transferred_obstacles": "Cumulated Number of Obstacles Transfers",
#         "cumulated_number_of_failed_goals": "Cumulated Number of Failed Goals"
#     }
#
#     if not plot_relevant_only:
#         # Plot each criterion individually
#         for criterion_id, criterion_name in criterion_id_to_criterion_name.items():
#             plot_criterion(
#                 namo_data, snamo_data, criterion_id, criterion_name,
#                 nb_goals, save_filepath=os.path.join(saved_plots_path, criterion_id+".html"),
#                 show=show, padding_percentage=padding_percentage
#             )
#
#     # Plot relevant criteria into single figure
#     plot_relevant_criteria(
#         namo_data, snamo_data, criterion_id_to_criterion_name, nb_goals=nb_goals,
#         save_filepath=os.path.join(saved_plots_path, "all_criteria.html"),
#         show=show, padding_percentage=padding_percentage
#     )
#
#
# def get_problematic_scenarios_ids(nb_goals, namo_logs_folder, snamo_logs_folder, nb_failures_max=50):
#     # Get all scenarios individual recorded data
#     namo_scenarios_ids = {
#         name for name in os.listdir(namo_logs_folder) if os.path.isdir(os.path.join(namo_logs_folder, name))
#     }
#     snamo_scenarios_ids = {
#         name for name in os.listdir(snamo_logs_folder) if os.path.isdir(os.path.join(snamo_logs_folder, name))
#     }
#
#     # Only keep scenarios for which we have both NAMO and SNAMO data
#     common_scenarios_ids = namo_scenarios_ids.union(snamo_scenarios_ids)
#
#     # For each scenario, aggregate the data of all goals so that cumulated data can be computed
#     namo_aggregated_goals_per_scenario_data = {}
#     snamo_aggregated_goals_per_scenario_data = {}
#     x = range(nb_goals)
#
#     namo_scenarios_without_results_file = []
#     snamo_scenarios_without_results_file = []
#
#     namo_scenarios_with_empty_results_file = []
#     snamo_scenarios_with_empty_results_file = []
#
#     namo_scenarios_with_exceptions = []
#     snamo_scenarios_with_exceptions = []
#
#     namo_scenarios_with_more_than_50_failed_goals = []
#     snamo_scenarios_with_more_than_50_failed_goals = []
#
#     for scenario_id in common_scenarios_ids:
#         namo_path = os.path.join(namo_logs_folder, scenario_id, "sim_results.json")
#         snamo_path = os.path.join(snamo_logs_folder, scenario_id, "sim_results.json")
#
#         try:
#             with open(namo_path, "r") as namo_f:
#                 namo_data = json.load(namo_f)
#
#             if "Exceptions" in namo_data:
#                 namo_scenarios_with_exceptions.append(scenario_id)
#
#             nb_failure_namo = len(
#                 [True for gr in namo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
#             if nb_failure_namo > nb_failures_max:
#                 namo_scenarios_with_more_than_50_failed_goals.append(scenario_id)
#         except Exception as e:
#             if isinstance(e, IOError):
#                 namo_scenarios_without_results_file.append(scenario_id)
#             if isinstance(e, ValueError):
#                 namo_scenarios_with_empty_results_file.append(scenario_id)
#
#         try:
#             with open(snamo_path, "r") as snamo_f:
#                 snamo_data = json.load(snamo_f)
#             if "Exceptions" in snamo_data:
#                 snamo_scenarios_with_exceptions.append(scenario_id)
#
#             nb_failure_snamo = len(
#                 [True for gr in snamo_data["agents"][0]["goals_reports"] if gr["goal_status"] == "failure"])
#             if nb_failure_snamo > nb_failures_max:
#                 snamo_scenarios_with_more_than_50_failed_goals.append(scenario_id)
#         except Exception as e:
#             if isinstance(e, IOError):
#                 snamo_scenarios_without_results_file.append(scenario_id)
#             if isinstance(e, ValueError):
#                 snamo_scenarios_with_empty_results_file.append(scenario_id)
#
#     return {
#         "namo_scenarios_without_results_file": namo_scenarios_without_results_file,
#         "snamo_scenarios_without_results_file": snamo_scenarios_without_results_file,
#         "namo_scenarios_with_exceptions": namo_scenarios_with_exceptions,
#         "snamo_scenarios_with_exceptions": snamo_scenarios_with_exceptions,
#         "namo_scenarios_with_more_than_50_failed_goals": namo_scenarios_with_more_than_50_failed_goals,
#         "snamo_scenarios_with_more_than_50_failed_goals": snamo_scenarios_with_more_than_50_failed_goals
#     }
#
#
# if __name__ == "__main__":
#     nb_goals = 200
#     main_dirname = os.path.join(os.path.dirname(__file__), "../../logs/04_after_the_feast/")
#
#     # Aggregate and plot data
#     # try:
#     #     with open(os.path.join(main_dirname, "synthesis.json"), "r") as f:
#     #         aggregated_data = json.load(f)
#     # except IOError as e:
#     #     namo_logs_folder = os.path.join(main_dirname, "stilman_2005_behavior_complexified_random_goal_no_reset/")
#     #     snamo_logs_folder = os.path.join(main_dirname, "stilman_2005_behavior_complexified_random_goal_no_reset_snamo/")
#     #     aggregated_data = aggregate_scenarios(nb_goals, namo_logs_folder, snamo_logs_folder, nb_failures_max=50)
#     #     with open(os.path.join(main_dirname, "synthesis.json"), "w") as f:
#     #         json.dump(aggregated_data, f)
#     #
#     # plot_data(aggregated_data, main_dirname, nb_goals=50, show=False, padding_percentage=0.05, plot_relevant_only=True)
#
#     # Extract problems in data
#     try:
#         with open(os.path.join(main_dirname, "problems.json"), "r") as f:
#             pb_data = json.load(f)
#     except IOError as e:
#         namo_logs_folder = os.path.join(main_dirname, "stilman_2005_behavior_complexified_random_goal_no_reset/")
#         snamo_logs_folder = os.path.join(main_dirname, "stilman_2005_behavior_complexified_random_goal_no_reset_snamo/")
#         pb_data = get_problematic_scenarios_ids(nb_goals, namo_logs_folder, snamo_logs_folder, nb_failures_max=50)
#         with open(os.path.join(main_dirname, "problems.json"), "w") as f:
#             json.dump(pb_data, f)


def zip_statistics(simulations_results_paths):
    # First pass to get max number of steps in all simulations that we will use as baseline to complete data after
    max_steps = 0
    for simulation_result_path in simulations_results_paths:
        try:
            with open(simulation_result_path, "r") as f:
                try:
                    sim_results = json.load(f)
                    max_steps = max(max_steps, len(sim_results['stats']))
                except Exception as e:
                    pass
        except IOError as e:
            pass

    # Second pass to actually zip all simulations stats together
    zipped_statistics = [[] for i in range(max_steps)]
    for simulation_result_path in simulations_results_paths:
        try:
            with open(simulation_result_path, "r") as f:
                try:
                    sim_results = json.load(f)
                    last_index, last_stepstats = 0, sim_results['stats'][0]
                    for index, stepstats in enumerate(sim_results['stats']):
                        zipped_statistics[index].append(stepstats)
                        last_index, last_stepstats = index, stepstats
                    nb_steps_to_replicate = max_steps - len(sim_results['stats'])
                    for i in range(nb_steps_to_replicate):
                        zipped_statistics[last_index + i + 1].append(last_stepstats)
                except Exception:
                    pass
        except IOError:
            pass
    return zipped_statistics


def aggregate_statistics(zipped_statistics):
    # Generate stats
    aggregated_stats = []
    aggregated_stats_by_agent = {uid: [] for uid in zipped_statistics[0][0]['agents_stats'].keys()}
    for index, step_stats_list in enumerate(zipped_statistics):
        # Aggregate stats over all agents in all simulations for this step
        aggregated_step_stats = {
            'max': StepStats(act_time=max(step_stats['act_time'] for step_stats in step_stats_list)),
            'sum': StepStats(act_time=sum(step_stats['act_time'] for step_stats in step_stats_list)),
            'avg': StepStats(act_time=np.average([step_stats['act_time'] for step_stats in step_stats_list])),
            'med': StepStats(act_time=np.median([step_stats['act_time'] for step_stats in step_stats_list]))
        }

        agents_stats_accross_simulations = []
        stats_per_agent_accross_simulations = {uid: [] for uid in aggregated_stats_by_agent.keys()}
        for step_stats in step_stats_list:
            for agent_uid, agent_stats in step_stats['agents_stats'].items():
                stats_per_agent_accross_simulations[agent_uid].append(agent_stats)
                agents_stats_accross_simulations.append(agent_stats)

        for criterion in AgentStepStats().__dict__.keys():
            setattr(aggregated_step_stats['max'].agents_stats, criterion, max([agent_stats[criterion] for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['sum'].agents_stats, criterion, sum([agent_stats[criterion] for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['avg'].agents_stats, criterion, np.average([agent_stats[criterion] for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['med'].agents_stats, criterion, np.median([agent_stats[criterion] for agent_stats in agents_stats_accross_simulations]))

        for criterion in WorldStepStats().__dict__.keys():
            setattr(aggregated_step_stats['max'].world_stats, criterion, max([step_stats['world_stats'][criterion] for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['sum'].world_stats, criterion, sum([step_stats['world_stats'][criterion] for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['avg'].world_stats, criterion, np.average([step_stats['world_stats'][criterion] for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['med'].world_stats, criterion, np.median([step_stats['world_stats'][criterion] for step_stats in step_stats_list]))

        aggregated_stats.append(aggregated_step_stats)

    return aggregated_stats


# def aggregate_statistics_by_agent(zipped_statistics):
#     # May or may not come in handy
#     # Generate stats
#     aggregated_stats_by_agent = {uid: [] for uid in zipped_statistics[0][0]['agents_stats'].keys()}
#     for index, step_stats_list in enumerate(zipped_statistics):
#         stats_per_agent_accross_simulations = {uid: [] for uid in aggregated_stats_by_agent.keys()}
#         for step_stats in step_stats_list:
#             for agent_uid, agent_stats in step_stats['agents_stats'].items():
#                 stats_per_agent_accross_simulations[agent_uid].append(agent_stats)
#
#         # Aggregate stats for each agent in all simulations for this step
#         for agent_uid, all_agent_stats_in_step in stats_per_agent_accross_simulations.items():
#             aggregated_step_agents_stats = {
#                 'max': AgentStepStats(),
#                 'sum': AgentStepStats(),
#                 'avg': AgentStepStats(),
#                 'med': AgentStepStats()
#             }
#
#             for key in AgentStepStats().__dict__.keys():
#                 setattr(aggregated_step_agents_stats['max'], key,
#                         max(agent_stats[key] for agent_stats in stats_per_agent_accross_simulations[agent_uid]))
#                 setattr(aggregated_step_agents_stats['sum'], key,
#                         sum(agent_stats[key] for agent_stats in stats_per_agent_accross_simulations[agent_uid]))
#                 setattr(aggregated_step_agents_stats['avg'], key,
#                         np.average([agent_stats[key] for agent_stats in stats_per_agent_accross_simulations[agent_uid]]))
#                 setattr(aggregated_step_agents_stats['med'], key,
#                         np.median([agent_stats[key] for agent_stats in stats_per_agent_accross_simulations[agent_uid]]))
#
#             aggregated_stats_by_agent[agent_uid].append(aggregated_step_agents_stats)
#
#     return aggregated_stats_by_agent


def plot_criterion(y, color="blue", dash=None):
    return go.Scatter(y=y, line=dict(color=color, dash=dash))


def scatter_plots_from_aggregated_statistics(aggregated_stats, color="blue", dash=None):
    aggregated_plots = {
        "max": StepStats(act_time=go.Scatter(y=[stats['max'].act_time for stats in aggregated_stats], line=dict(color=color, dash=dash))),
        "sum": StepStats(act_time=go.Scatter(y=[stats['sum'].act_time for stats in aggregated_stats], line=dict(color=color, dash=dash))),
        "avg": StepStats(act_time=go.Scatter(y=[stats['avg'].act_time for stats in aggregated_stats], line=dict(color=color, dash=dash))),
        "med": StepStats(act_time=go.Scatter(y=[stats['med'].act_time for stats in aggregated_stats], line=dict(color=color, dash=dash)))
    }

    for criterion in AgentStepStats().__dict__.keys():
        setattr(aggregated_plots['max'].agents_stats, criterion, go.Scatter(y=[getattr(stats['max'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))
        setattr(aggregated_plots['sum'].agents_stats, criterion, go.Scatter(y=[getattr(stats['sum'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))
        setattr(aggregated_plots['avg'].agents_stats, criterion, go.Scatter(y=[getattr(stats['avg'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))
        setattr(aggregated_plots['med'].agents_stats, criterion, go.Scatter(y=[getattr(stats['med'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))

    for criterion in WorldStepStats().__dict__.keys():
        setattr(aggregated_plots['max'].world_stats, criterion, go.Scatter(y=[getattr(stats['max'].world_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))
        setattr(aggregated_plots['sum'].world_stats, criterion, go.Scatter(y=[getattr(stats['sum'].world_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))
        setattr(aggregated_plots['avg'].world_stats, criterion, go.Scatter(y=[getattr(stats['avg'].world_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))
        setattr(aggregated_plots['med'].world_stats, criterion, go.Scatter(y=[getattr(stats['med'].world_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash)))

    return aggregated_plots

namo_sim_results_paths = ['/home/xia0ben/INRIA/Code/s-namo-sim/logs/04_after_the_feast/stilman_2005_behavior_multi_robots_complexified/2021-07-21-10h42m29s_616116/sim_results.json']
snamo_sim_results_paths = ['/home/xia0ben/INRIA/Code/s-namo-sim/logs/04_after_the_feast/stilman_2005_behavior_multi_robots_complexified_snamo/2021-07-21-10h52m46s_708636/sim_results.json']

if __name__ == '__main__':
    # Command to clean up JSON logs from Infinite values to "Infinite" ones and allow parsing by browser
    # find ./ -name 'sim_results.json' -exec sed -i 's/""Infinity""/"Infinity"/g' {} \;

    # MAIN_FOLDER = "/home/xia0ben/logs2/logs/"
    # scenarios_ids = {
    #     name for name in os.listdir(MAIN_FOLDER) if os.path.isdir(os.path.join(MAIN_FOLDER, name))
    # }
    #
    # scenario_paths_with_exceptions = []
    # scenario_paths_without_exceptions = []
    #
    # for scenario_id in scenarios_ids:
    #     namo_folder = os.path.join(MAIN_FOLDER, scenario_id, "sim_namo_" + scenario_id)
    #     snamo_folder = os.path.join(MAIN_FOLDER, scenario_id, "sim_snamo_" + scenario_id)
    #
    #     try:
    #         namo_scenarios_folders = {
    #             name for name in os.listdir(namo_folder) if os.path.isdir(os.path.join(namo_folder, name))
    #         }
    #
    #         for namo_scenario_folder in namo_scenarios_folders:
    #             scenario_path = os.path.join(namo_folder, namo_scenario_folder, "sim_results.json")
    #             try:
    #                 with open(scenario_path, "r") as f:
    #                     sim_results = json.load(f)
    #                     if sim_results.get("Exceptions", None):
    #                         scenario_paths_with_exceptions.append(scenario_path)
    #                     else:
    #                         if sim_results:
    #                             scenario_paths_without_exceptions.append(scenario_path)
    #             except Exception as e:
    #                 pass
    #
    #         # snamo_scenarios_folders = {
    #         #     name for name in os.listdir(snamo_folder) if os.path.isdir(os.path.join(snamo_folder, name))
    #         # }
    #     except Exception as e:
    #         pass
    #
    # nb_scenarios_without_exceptions = len(scenario_paths_without_exceptions)
    # total_nb_scenarios = len(scenario_paths_with_exceptions) + len(scenario_paths_without_exceptions)
    #
    # print("{} over {} scenarios were executed without exceptions.".format(nb_scenarios_without_exceptions, total_nb_scenarios))

    namo_sim_results_zipped_statistics = zip_statistics(namo_sim_results_paths)
    namo_sim_results_aggregated_statistics = aggregate_statistics(namo_sim_results_zipped_statistics)
    namo_scatter_plots = scatter_plots_from_aggregated_statistics(namo_sim_results_aggregated_statistics)

    snamo_sim_results_zipped_statistics = zip_statistics(snamo_sim_results_paths)
    snamo_sim_results_aggregated_statistics = aggregate_statistics(snamo_sim_results_zipped_statistics)
    snamo_scatter_plots = scatter_plots_from_aggregated_statistics(snamo_sim_results_aggregated_statistics)

    print('')

