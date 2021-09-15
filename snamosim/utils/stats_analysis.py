import os
import json
import pickle
import time
import multiprocessing
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


def get_max_nb_steps_multiprocessing(simulations_results_paths, return_list):
    max_steps = get_max_nb_steps(simulations_results_paths)
    return_list.append(max_steps)


def zip_statistics(scenarios_stats_paths, max_steps, start=0, stop=None):
    if stop is None:
        stop = max_steps
    zipped_statistics = [[] for i in range(stop-start)]

    for scenario_stats_path in scenarios_stats_paths:
        try:
            with open(scenario_stats_path, "rb") as f:
                try:
                    stats = pickle.load(f)['stats']
                    last_stepstats = stats[-1]

                    for counter, index in enumerate(range(start, stop)):
                        if index < len(stats):
                            stepstats = stats[index]
                            zipped_statistics[counter].append(stepstats)
                        else:
                            zipped_statistics[counter].append(last_stepstats)
                except Exception as e:
                    pass
        except IOError as e:
            pass
    return zipped_statistics


def zip_initial_and_final_statistics(scenarios_stats_paths):
    initial_stats, final_stats = [], []

    for scenario_stats_path in scenarios_stats_paths:
        try:
            with open(scenario_stats_path, "rb") as f:
                try:
                    stats = pickle.load(f)['stats']
                    init, final = stats[0], stats[-1]
                    initial_stats.append(init)
                    final_stats.append(final)
                except Exception as e:
                    pass
        except IOError as e:
            pass
    return [initial_stats, final_stats]


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


def aggregate_statistics_multiprocessing(scenarios_stats_paths, max_steps, start, stop, return_dict, key):
    return_dict[key] = aggregate_statistics(zip_statistics(scenarios_stats_paths, max_steps, start, stop))


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
                showlegend=False, y=upper_criterion
            ),
            go.Scatter(
                name='Lower', marker=dict(color=color), line=dict(width=0), mode='lines',
                fillcolor=fillcolor, fill='tonexty', showlegend=False, y=lower_criterion
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
                showlegend=False, y=upper_criterion
            ),
            go.Scatter(
                name='Lower', marker=dict(color=color), line=dict(width=0), mode='lines',
                fillcolor=fillcolor, fill='tonexty', showlegend=False, y=lower_criterion
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
    nb_cpu = multiprocessing.cpu_count()
    nb_usable_cpus = 4  # nb_cpu - 1

    manager = multiprocessing.Manager()

    nb_steps_per_aggregation_op_2 = 1000
    nb_steps_per_aggregation_op_1 = 1000

    MAIN_FOLDER = "/home/xia0ben/INRIA/Code/s-namo-sim/logs/citi_2r_50g"
    scenarios_ids = {
        name for name in os.listdir(MAIN_FOLDER) if os.path.isdir(os.path.join(MAIN_FOLDER, name))
    }

    paths_with_exceptions = []
    paths_without_exceptions = []

    namo_sim_results_paths, snamo_sim_results_paths = [], []

    print('----------------------------------------------------------')
    print('Loading scenarios stats and displaying problematic files :')
    print('----------------------------------------------------------')

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

    print('----------------------------------------------------------')
    print('Zipping, aggregating and saving NAMO + S-NAMO results in initial and final state:')
    print('----------------------------------------------------------')

    namo_init_final_stats = aggregate_statistics(zip_initial_and_final_statistics(namo_sim_results_paths))
    snamo_init_final_stats = aggregate_statistics(zip_initial_and_final_statistics(snamo_sim_results_paths))

    namo_init_avg, namo_final_avg = namo_init_final_stats[0]['avg'], namo_init_final_stats[1]['avg']
    snamo_init_avg, snamo_final_avg = snamo_init_final_stats[0]['avg'], snamo_init_final_stats[1]['avg']

    namo_init_std, namo_final_std = namo_init_final_stats[0]['std'], namo_init_final_stats[1]['std']
    snamo_init_std, snamo_final_std = snamo_init_final_stats[0]['std'], snamo_init_final_stats[1]['std']

    print("Scenario                                                             & Conflicts           & R-R                 &  R-O                &  S-O                & Unpostponements                           & Recomputations         & Wait Steps            & Successes                    & $L_{transfer}$                                          & Transfers           & $T_{planning}$    \\\\")
    print("                                                                     &                     & Conflicts           &  Conflicts          &  Conflicts          & / Postponements                           &                        & / Total Steps         & / Goals                      & / $L_{total}$                                           &                     & (s)               \\\\")
    print("                                                                     &                     &                     &                     &                     &                                           &                        &                       &                              & (m) / (m)                                               &                     &                   \\\\ \hline")

    print("AtF - 2                                                              &                     &                     &                     &                     &                                           &                        &                       &                              &                                                         &                     &                   \\\\")
    print("C-NAMO                                                               & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} / {:.0f} $\pm$ {:.0f} &  {:.0f} $\pm$ {:.0f}   &  {:.0f} $\pm$ {:.0f}  & {:.0f} $\pm$ {:.0f} / {:.0f} & {:.1f} $\pm$ {:.1f} / {:.1f} $\pm$ {:.1f}               & {:.0f} $\pm$ {:.0f} & {:.1f} $\pm$ {:.1f} \\\\".format(
        namo_final_avg.agents_stats.nb_conflicts, namo_final_std.agents_stats.nb_conflicts,
        namo_final_avg.agents_stats.nb_robot_robot_conflicts, namo_final_std.agents_stats.nb_robot_robot_conflicts,
        namo_final_avg.agents_stats.nb_robot_obstacle_conflicts, namo_final_std.agents_stats.nb_robot_obstacle_conflicts,
        namo_final_avg.agents_stats.nb_stolen_movable_conflicts, namo_final_std.agents_stats.nb_stolen_movable_conflicts,
        namo_final_avg.agents_stats.nb_of_unpostponements, namo_final_std.agents_stats.nb_of_unpostponements, namo_final_avg.agents_stats.nb_of_postponements, namo_final_std.agents_stats.nb_of_postponements,
        namo_final_avg.agents_stats.nb_of_plan_computations, namo_final_std.agents_stats.nb_of_plan_computations,
        namo_final_avg.agents_stats.nb_wait_steps, namo_final_std.agents_stats.nb_wait_steps,  # ADD TOTAL NUMBER OF STEPS METRIC ONCE ITS BEEN COMPUTED
        namo_final_avg.agents_stats.nb_successful_goals, namo_final_std.agents_stats.nb_successful_goals, namo_final_avg.agents_stats.nb_goals,
        namo_final_avg.agents_stats.transfer_path_length, namo_final_std.agents_stats.transfer_path_length, namo_final_avg.agents_stats.path_length, namo_final_std.agents_stats.path_length,
        namo_final_avg.agents_stats.nb_transfers, namo_final_std.agents_stats.nb_transfers,
        namo_final_avg.agents_stats.think_time, namo_final_std.agents_stats.think_time
    ))
    print("SC-NAMO                                                              & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} & {:.0f} $\pm$ {:.0f} / {:.0f} $\pm$ {:.0f} &  {:.0f} $\pm$ {:.0f}   &  {:.0f} $\pm$ {:.0f}  & {:.0f} $\pm$ {:.0f} / {:.0f} & {:.1f} $\pm$ {:.1f} / {:.1f} $\pm$ {:.1f}               & {:.0f} $\pm$ {:.0f} & {:.1f} $\pm$ {:.1f} \\\\ \hline".format(
        snamo_final_avg.agents_stats.nb_conflicts, snamo_final_std.agents_stats.nb_conflicts,
        snamo_final_avg.agents_stats.nb_robot_robot_conflicts, snamo_final_std.agents_stats.nb_robot_robot_conflicts,
        snamo_final_avg.agents_stats.nb_robot_obstacle_conflicts, snamo_final_std.agents_stats.nb_robot_obstacle_conflicts,
        snamo_final_avg.agents_stats.nb_stolen_movable_conflicts, snamo_final_std.agents_stats.nb_stolen_movable_conflicts,
        snamo_final_avg.agents_stats.nb_of_unpostponements, snamo_final_std.agents_stats.nb_of_unpostponements, snamo_final_avg.agents_stats.nb_of_postponements, snamo_final_std.agents_stats.nb_of_postponements,
        snamo_final_avg.agents_stats.nb_of_plan_computations, snamo_final_std.agents_stats.nb_of_plan_computations,
        snamo_final_avg.agents_stats.nb_wait_steps, snamo_final_std.agents_stats.nb_wait_steps,  # ADD TOTAL NUMBER OF STEPS METRIC ONCE ITS BEEN COMPUTED
        snamo_final_avg.agents_stats.nb_successful_goals, snamo_final_std.agents_stats.nb_successful_goals, snamo_final_avg.agents_stats.nb_goals,
        snamo_final_avg.agents_stats.transfer_path_length, snamo_final_std.agents_stats.transfer_path_length, snamo_final_avg.agents_stats.path_length, snamo_final_std.agents_stats.path_length,
        snamo_final_avg.agents_stats.nb_transfers, snamo_final_std.agents_stats.nb_transfers,
        snamo_final_avg.agents_stats.think_time, snamo_final_std.agents_stats.think_time
    ))

    print("Scenario                                                             & $Ncc(W^{t_{init}})$          & $C^{acc}_{h}(W^{t_{init}})$          & $ST(W^{t_{init}})$  \\\\")
    print("                                                                     & / $Ncc(W^{t_{end}})$         & / $C^{acc}_{h}(W^{t_{end}})$         & / $ST(W^{t_{end}})$ \\\\ \hline")

    print("AtF - 2                                                              & {:.0f}                       & {:.0f}                               & {:.0f}                 \\\\".format(
        namo_init_avg.world_stats.nb_components,
        namo_init_avg.world_stats.free_space_size,
        namo_init_avg.world_stats.absolute_social_cost
    ))
    print("C-NAMO                                                               & / {:.1f} $\pm$ {:.1f}        & / {:.0f} $\pm$ {:.0f}                & / {:.0f} $\pm$ {:.0f}      \\\\".format(
        namo_final_avg.world_stats.nb_components, namo_final_std.world_stats.nb_components,
        namo_final_avg.world_stats.free_space_size, namo_final_std.world_stats.free_space_size,
        namo_final_avg.world_stats.absolute_social_cost, namo_final_std.world_stats.absolute_social_cost
    ))
    print("SC-NAMO                                                              & / {:.1f} $\pm$ {:.1f}        & / {:.0f} $\pm$ {:.0f}                & / {:.0f} $\pm$ {:.0f}      \\\\ \hline".format(
        snamo_final_avg.world_stats.nb_components, snamo_final_std.world_stats.nb_components,
        snamo_final_avg.world_stats.free_space_size, snamo_final_std.world_stats.free_space_size,
        snamo_final_avg.world_stats.absolute_social_cost, snamo_final_std.world_stats.absolute_social_cost
    ))


    all_paths = namo_sim_results_paths + snamo_sim_results_paths
    print('----------------------------------------------------------')
    print('Computing max number of steps accross all {} scenarios...'.format(len(all_paths)))
    print('----------------------------------------------------------')

    paths_per_process= len(all_paths) // nb_usable_cpus
    left_paths_nb = len(all_paths) % nb_usable_cpus
    max_nb_steps_list = manager.list()
    processes = []
    last_index = 0
    for i in range(nb_usable_cpus):
        start, stop = last_index, last_index + paths_per_process
        if left_paths_nb > 0:
            stop += 1
            left_paths_nb -= 1
        last_index = stop
        paths_for_process = all_paths[start:stop]
        process = multiprocessing.Process(
            target=get_max_nb_steps_multiprocessing, args=(paths_for_process, max_nb_steps_list)
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

    max_steps = max(max_nb_steps_list)

    print('----------------------------------------------------------')
    print('Max number of steps is {}.'.format(max_steps))
    print('----------------------------------------------------------')

    print('----------------------------------------------------------')
    print('Zipping and aggregating S-NAMO results by packs of {}:'.format(nb_steps_per_aggregation_op_2))
    print('----------------------------------------------------------')

    # time_start_2 = time.time()
    # snamo_aggregated_stats = []
    # for i in range(0, max_steps, nb_steps_per_aggregation_op_2):
    #     start, stop = i, i + nb_steps_per_aggregation_op_2
    #     stop = max_steps if stop > max_steps else stop
    #     print('Aggregating stats from steps {} to {}...'.format(start, stop-1))
    #     snamo_aggregated_stats += aggregate_statistics(zip_statistics(snamo_sim_results_paths, max_steps, start, stop))
    #     print('Aggregated stats from steps {} to {}.'.format(start, stop-1))
    # time_2 = time.time() - time_start_2

    time_start_1 = time.time()
    processes = []
    snamo_aggregated_stats_dict = manager.dict()
    last_index = 0
    counter = 0
    while last_index < max_steps:
        if len(processes) < nb_usable_cpus:
            start, stop = last_index, last_index + nb_steps_per_aggregation_op_1
            stop = max_steps if stop > max_steps else stop
            last_index = stop
            print('Aggregating stats from steps {} to {}...'.format(start, stop - 1))
            process = multiprocessing.Process(
                target=aggregate_statistics_multiprocessing,
                args=(snamo_sim_results_paths, max_steps, start, stop, snamo_aggregated_stats_dict, counter)
            )
            counter += 1
            processes.append(process)
            process.start()
        else:
            for index, process in enumerate(processes):
                if not process.is_alive():
                    process.terminate()
                    del processes[index]
            time.sleep(1.)
    for process in processes:
        process.join()
    snamo_aggregated_stats = [stepstats for key in range(counter) for stepstats in snamo_aggregated_stats_dict[key]]
    time_1 = time.time() - time_start_1

    print('----------------------------------------------------------')
    print('Zipping and aggregating NAMO results by packs of {}:'.format(nb_steps_per_aggregation_op_1))
    print('----------------------------------------------------------')

    time_start_2 = time.time()
    processes = []
    namo_aggregated_stats_dict = manager.dict()
    last_index = 0
    counter = 0
    while last_index < max_steps:
        if len(processes) < nb_usable_cpus:
            start, stop = last_index, last_index + nb_steps_per_aggregation_op_1
            stop = max_steps if stop > max_steps else stop
            last_index = stop
            print('Aggregating stats from steps {} to {}...'.format(start, stop-1))
            process = multiprocessing.Process(
                target=aggregate_statistics_multiprocessing,
                args=(namo_sim_results_paths, max_steps, start, stop, namo_aggregated_stats_dict, counter)
            )
            counter += 1
            processes.append(process)
            process.start()
        else:
            for index, process in enumerate(processes):
                if not process.is_alive():
                    process.terminate()
                    del processes[index]
            time.sleep(1.)
    for process in processes:
        process.join()
    namo_aggregated_stats = [stepstats for key in range(counter) for stepstats in namo_aggregated_stats_dict[key]]
    time_2 = time.time() - time_start_2

    print('time_1: {}, time_2: {}'.format(time_1, time_2))

    print('----------------------------------------------------------')
    print('Aggregation completed. Generating plots...')
    print('----------------------------------------------------------')

    namo_operations_to_scatter_plots = scatter_plots_from_aggregated_statistics(namo_aggregated_stats, name='NAMO')

    if snamo_sim_results_paths:
        snamo_operations_to_scatter_plots = scatter_plots_from_aggregated_statistics(snamo_aggregated_stats, color='green', fillcolor="rgba(0, 128, 0, .1)", name='S-NAMO')

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

    print('----------------------------------------------------------')
    print('Generated plots.')
    print('----------------------------------------------------------')