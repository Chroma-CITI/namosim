import os
import json
import pickle
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from snamosim.simulator import AgentStepStats, WorldStepStats, StepStats


def get_max_nb_steps(simulations_results_paths):
    max_steps = 0
    for simulation_result_path in simulations_results_paths:
        try:
            with open(simulation_result_path, "rb") as f:
                try:
                    sim_results = pickle.load(f)
                    max_steps = max(max_steps, len(sim_results['stats']))
                except Exception as e:
                    pass
        except IOError as e:
            pass
    return max_steps


def zip_statistics(simulations_results_paths, max_steps):
    zipped_statistics = [[] for i in range(max_steps)]
    for simulation_result_path in simulations_results_paths:
        try:
            with open(simulation_result_path, "rb") as f:
                try:
                    sim_results = pickle.load(f)
                    last_index, last_stepstats = 0, sim_results['stats'][0]
                    for index, stepstats in enumerate(sim_results['stats']):
                        zipped_statistics[index].append(stepstats)
                        last_index, last_stepstats = index, stepstats
                    nb_steps_to_replicate = max_steps - len(sim_results['stats'])
                    for i in range(nb_steps_to_replicate):
                        zipped_statistics[last_index + i + 1].append(last_stepstats)
                except Exception as e:
                    pass
        except IOError as e:
            pass
    return zipped_statistics


def aggregate_statistics_dict(zipped_statistics):
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


def aggregate_statistics(zipped_statistics, start_index=0, end_index=None):
    if end_index is None:
        end_index = len(zipped_statistics)
    aggregated_stats = []
    for index in range(start_index, end_index):
        step_stats_list = zipped_statistics[index]

        # Aggregate stats over all agents in all simulations for this step
        aggregated_step_stats = {
            'min': StepStats(act_time=min(step_stats.act_time for step_stats in step_stats_list)),
            'max': StepStats(act_time=max(step_stats.act_time for step_stats in step_stats_list)),
            'med': StepStats(act_time=np.median([step_stats.act_time for step_stats in step_stats_list])),
            'q1': StepStats(act_time=np.quantile([step_stats.act_time for step_stats in step_stats_list], 0.25)),
            'q3': StepStats(act_time=np.quantile([step_stats.act_time for step_stats in step_stats_list], 0.75)),
            'sum': StepStats(act_time=sum(step_stats.act_time for step_stats in step_stats_list)),
            'avg': StepStats(act_time=np.average([step_stats.act_time for step_stats in step_stats_list])),
            'std': StepStats(act_time=np.std([step_stats.act_time for step_stats in step_stats_list]))
        }

        agents_stats_accross_simulations = []
        for step_stats in step_stats_list:
            for agent_uid, agent_stats in step_stats.agents_stats.items():
                agents_stats_accross_simulations.append(agent_stats)

        for criterion in AgentStepStats().__dict__.keys():
            setattr(aggregated_step_stats['min'].agents_stats, criterion, min([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['max'].agents_stats, criterion, max([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['med'].agents_stats, criterion, np.median([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['q1'].agents_stats, criterion, np.quantile([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations], 0.25))
            setattr(aggregated_step_stats['q3'].agents_stats, criterion, np.quantile([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations], 0.75))
            setattr(aggregated_step_stats['sum'].agents_stats, criterion, sum([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['avg'].agents_stats, criterion, np.average([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))
            setattr(aggregated_step_stats['std'].agents_stats, criterion, np.std([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))

        for criterion in WorldStepStats().__dict__.keys():
            setattr(aggregated_step_stats['min'].world_stats, criterion, min([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['max'].world_stats, criterion, max([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['med'].world_stats, criterion, np.median([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['q1'].world_stats, criterion, np.quantile([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list], 0.25))
            setattr(aggregated_step_stats['q3'].world_stats, criterion, np.quantile([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list], 0.75))
            setattr(aggregated_step_stats['sum'].world_stats, criterion, sum([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['avg'].world_stats, criterion, np.average([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))
            setattr(aggregated_step_stats['std'].world_stats, criterion, np.std([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))

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


def scatter_plots_from_aggregated_statistics(aggregated_stats, color="blue", fillcolor="rgba(0, 0, 255, .1)", dash=None, name=''):
    avg_act_time = np.array([stats['avg'].act_time for stats in aggregated_stats])
    std_act_time = np.array([stats['std'].act_time for stats in aggregated_stats])
    upper_act_time, lower_act_time = avg_act_time + std_act_time,  avg_act_time - std_act_time
    lower_act_time[lower_act_time < 0] = 0

    aggregated_plots = {
        "sum": StepStats(act_time=[
            go.Scatter(y=[stats['sum'].act_time for stats in aggregated_stats], line=dict(color=color, dash=dash), name=name)
        ]),
        "avg": StepStats(act_time=[
            go.Scatter(y=avg_act_time, line=dict(color=color, dash=dash), name=name, mode='lines'),
            go.Scatter(
                name='Upper', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=upper_act_time
            ),
            go.Scatter(
                name='Lower', marker=dict(color=color), line=dict(width=0), mode='lines',
                fillcolor=fillcolor, fill='tonexty', showlegend=False, y=lower_act_time
            )
        ]),
        "med": StepStats(act_time=[
            go.Scatter(y=[stats['med'].act_time for stats in aggregated_stats], line=dict(color=color, dash=dash), name=name, mode='lines'),
            go.Scatter(
                name='Max', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=[stats['max'].act_time for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Min', marker=dict(color=color), line=dict(width=0), mode='lines', fillcolor=fillcolor,
                fill='tonexty', showlegend=False, y=[stats['min'].act_time for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Q1', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=[stats['q1'].act_time for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Q3', marker=dict(color=color), line=dict(width=0), mode='lines', fillcolor=fillcolor,
                fill='tonexty', showlegend=False, y=[stats['q3'].act_time for stats in aggregated_stats]
            )
        ])
    }

    for criterion in AgentStepStats().__dict__.keys():
        avg_criterion = np.array([getattr(stats['avg'].agents_stats, criterion) for stats in aggregated_stats])
        std_criterion = np.array([getattr(stats['std'].agents_stats, criterion) for stats in aggregated_stats])
        upper_criterion, lower_criterion = avg_criterion + std_criterion, avg_criterion - std_criterion
        lower_criterion[lower_criterion < 0] = 0

        setattr(aggregated_plots['sum'].agents_stats, criterion, [
            go.Scatter(y=[getattr(stats['sum'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash), name=name)
        ])
        setattr(aggregated_plots['avg'].agents_stats, criterion, [
            go.Scatter(y=[getattr(stats['avg'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash), name=name, mode='lines'),
            go.Scatter(
                name='Upper', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=avg_criterion + std_criterion
            ),
            go.Scatter(
                name='Lower', marker=dict(color=color), line=dict(width=0), mode='lines',
                fillcolor=fillcolor, fill='tonexty', showlegend=False, y=avg_criterion - std_criterion
            )
        ])
        setattr(aggregated_plots['med'].agents_stats, criterion, [
            go.Scatter(y=[getattr(stats['med'].agents_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash), name=name, mode='lines'),
            go.Scatter(
                name='Max', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=[getattr(stats['max'].agents_stats, criterion) for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Min', marker=dict(color=color), line=dict(width=0), mode='lines', fillcolor=fillcolor,
                fill='tonexty', showlegend=False, y=[getattr(stats['min'].agents_stats, criterion) for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Q1', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=[getattr(stats['q1'].agents_stats, criterion) for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Q3', marker=dict(color=color), line=dict(width=0), mode='lines', fillcolor=fillcolor,
                fill='tonexty', showlegend=False, y=[getattr(stats['q3'].agents_stats, criterion) for stats in aggregated_stats]
            )
        ])

    for criterion in WorldStepStats().__dict__.keys():
        avg_criterion = np.array([getattr(stats['avg'].world_stats, criterion) for stats in aggregated_stats])
        std_criterion = np.array([getattr(stats['std'].world_stats, criterion) for stats in aggregated_stats])
        upper_criterion, lower_criterion = avg_criterion + std_criterion, avg_criterion - std_criterion
        lower_criterion[lower_criterion < 0] = 0

        setattr(aggregated_plots['sum'].world_stats, criterion, [
            go.Scatter(y=[getattr(stats['sum'].world_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash))
        ])
        setattr(aggregated_plots['avg'].world_stats, criterion, [
            go.Scatter(y=avg_criterion, line=dict(color=color, dash=dash), name=name, mode='lines'),
            go.Scatter(
                name='Upper', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=avg_criterion + std_criterion
            ),
            go.Scatter(
                name='Lower', marker=dict(color=color), line=dict(width=0), mode='lines',
                fillcolor=fillcolor, fill='tonexty', showlegend=False, y=avg_criterion - std_criterion
            )
        ])
        setattr(aggregated_plots['med'].world_stats, criterion, [
            go.Scatter(y=[getattr(stats['med'].world_stats, criterion) for stats in aggregated_stats], line=dict(color=color, dash=dash), name=name, mode='lines'),
            go.Scatter(
                name='Max', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=[getattr(stats['max'].world_stats, criterion) for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Min', marker=dict(color=color), line=dict(width=0), mode='lines', fillcolor=fillcolor,
                fill='tonexty', showlegend=False, y=[getattr(stats['min'].world_stats, criterion) for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Q1', mode='lines', marker=dict(color=color), line=dict(width=0),
                showlegend=False, y=[getattr(stats['q1'].world_stats, criterion) for stats in aggregated_stats]
            ),
            go.Scatter(
                name='Q3', marker=dict(color=color), line=dict(width=0), mode='lines', fillcolor=fillcolor,
                fill='tonexty', showlegend=False, y=[getattr(stats['q3'].world_stats, criterion) for stats in aggregated_stats]
            )
        ])

    return aggregated_plots


if __name__ == '__main__':
    # Command to clean up JSON logs from Infinite values to "Infinite" ones and allow parsing by browser
    # find ./ -name 'sim_results.json' -exec sed -i 's/""Infinity""/"Infinity"/g' {} \;

    MAIN_FOLDER = "/home/xia0ben/INRIA/Code/s-namo-sim/logs_1000_full/"
    scenarios_ids = {
        name for name in os.listdir(MAIN_FOLDER) if os.path.isdir(os.path.join(MAIN_FOLDER, name))
    }

    paths_with_exceptions = []
    paths_without_exceptions = []

    namo_sim_results_paths, snamo_sim_results_paths = [], []

    for scenario_id in scenarios_ids:
        namo_folder = os.path.join(MAIN_FOLDER, scenario_id, "sim_namo_" + scenario_id)
        snamo_folder = os.path.join(MAIN_FOLDER, scenario_id, "sim_snamo_" + scenario_id)

        try:
            namo_scenarios_folders = [
                name for name in os.listdir(namo_folder) if os.path.isdir(os.path.join(namo_folder, name))
            ]
            snamo_scenarios_folders = [
                name for name in os.listdir(snamo_folder) if os.path.isdir(os.path.join(snamo_folder, name))
            ]

            if len(namo_scenarios_folders) > 1:
                print("Scenario {} has several run folders for NAMO: {}".format(scenario_id, namo_scenarios_folders))
            elif len(namo_scenarios_folders) == 0:
                print("Scenario {} has no run folders for NAMO.".format(scenario_id))

            if len(snamo_scenarios_folders) > 1:
                print("Scenario {} has several run folders for SNAMO: {}".format(scenario_id, snamo_scenarios_folders))
            elif len(snamo_scenarios_folders) == 0:
                print("Scenario {} has no run folders for SNAMO.".format(scenario_id))

            namo_stats_path = os.path.join(namo_folder, namo_scenarios_folders[0], "stats.pickle")
            snamo_stats_path = os.path.join(snamo_folder, snamo_scenarios_folders[0], "stats.pickle")

            with open(namo_stats_path, "rb") as f:
                pass
            with open(snamo_stats_path, "rb") as f:
                pass

            namo_sim_results_paths.append(namo_stats_path)
            snamo_sim_results_paths.append(snamo_stats_path)

        except Exception as e:
            print(e)

    ##############

    # namo_sim_results_paths = [
    #     '/home/xia0ben/INRIA/Code/s-namo-sim/logs/0000/sim_namo_0000/2021-08-24-12h09m04s_199363/stats.pickle'
    # ]
    # snamo_sim_results_paths = [
    #     '/home/xia0ben/INRIA/Code/s-namo-sim/logs/0000/sim_snamo_0000/2021-08-24-12h09m04s_199363/stats.pickle'
    # ]

    ##############

    max_steps = get_max_nb_steps(namo_sim_results_paths + snamo_sim_results_paths)

    # Zip statistics per simulation step
    namo_sim_results_zipped_statistics = zip_statistics(namo_sim_results_paths, max_steps)
    if snamo_sim_results_paths:
        snamo_sim_results_zipped_statistics = zip_statistics(snamo_sim_results_paths, max_steps)

    # Aggregate first and last step statistics for table
    first_namo_stats = aggregate_statistics(namo_sim_results_zipped_statistics, start_index=0, end_index=1)[0]
    last_namo_stats = aggregate_statistics(
        namo_sim_results_zipped_statistics,
        start_index=len(namo_sim_results_zipped_statistics)-1, end_index=len(namo_sim_results_zipped_statistics)
    )[0]

    if snamo_sim_results_paths:
        first_snamo_stats = aggregate_statistics(snamo_sim_results_zipped_statistics, start_index=0, end_index=1)[0]
        last_snamo_stats = aggregate_statistics(
            snamo_sim_results_zipped_statistics,
            start_index=len(snamo_sim_results_zipped_statistics) - 1, end_index=len(snamo_sim_results_zipped_statistics)
        )[0]

    for criterion in AgentStepStats().__dict__.keys():
        setattr(aggregated_step_stats['min'].agents_stats, criterion,
                min([getattr(agent_stats, criterion) for agent_stats in agents_stats_accross_simulations]))

    for criterion in WorldStepStats().__dict__.keys():
        setattr(aggregated_step_stats['min'].world_stats, criterion,
                min([getattr(step_stats.world_stats, criterion) for step_stats in step_stats_list]))

    # Aggregate statistics for plots
    namo_sim_results_aggregated_statistics = aggregate_statistics(namo_sim_results_zipped_statistics)
    namo_operations_to_scatter_plots = scatter_plots_from_aggregated_statistics(namo_sim_results_aggregated_statistics, name='NAMO')

    if snamo_sim_results_paths:
        snamo_sim_results_aggregated_statistics = aggregate_statistics(snamo_sim_results_zipped_statistics)
        snamo_operations_to_scatter_plots = scatter_plots_from_aggregated_statistics(snamo_sim_results_aggregated_statistics, color='green', fillcolor="rgba(0, 128, 0, .1)", name='S-NAMO')

    for aggregation_operation, namo_scatter_plots in namo_operations_to_scatter_plots.items():
        if snamo_sim_results_paths:
            snamo_scatter_plots = snamo_operations_to_scatter_plots[aggregation_operation]

        nb_criteria = 1 + len(AgentStepStats().__dict__) + len(WorldStepStats().__dict__)

        fig = sp.make_subplots(
            rows=nb_criteria, cols=1,
            subplot_titles=['act_time']+list(AgentStepStats().__dict__.keys())+list(WorldStepStats().__dict__.keys())
        )

        for plot in namo_scatter_plots.act_time:
            fig.append_trace(plot, row=1, col=1)
        if snamo_sim_results_paths:
            for plot in snamo_scatter_plots.act_time:
                fig.append_trace(plot, row=1, col=1)

        for index, criterion in enumerate(AgentStepStats().__dict__.keys()):
            for plot in getattr(namo_scatter_plots.agents_stats, criterion):
                fig.append_trace(plot, row=1+1+index, col=1)
            if snamo_sim_results_paths:
                for plot in getattr(snamo_scatter_plots.agents_stats, criterion):
                    fig.append_trace(plot, row=1+1+index, col=1)

        for index, criterion in enumerate(WorldStepStats().__dict__.keys()):
            for plot in getattr(namo_scatter_plots.world_stats, criterion):
                fig.append_trace(plot, row=1+1+len(AgentStepStats().__dict__.keys())+index, col=1)
            if snamo_sim_results_paths:
                for plot in getattr(snamo_scatter_plots.world_stats, criterion):
                    fig.append_trace(plot, row=1+1+len(AgentStepStats().__dict__.keys())+index, col=1)

        fig.update_layout(height=12000, title_text=aggregation_operation, showlegend=False, hovermode="x")
        fig.show()
