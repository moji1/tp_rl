#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ0 - Variation of scheduling times
from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = True
RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True
scheduling_time_ratios = np.arange(0.1, 1, 0.1)


def exp_scheduling_time(iteration):
    avg_napfd = []

    ags = [
        lambda: (
            agents.NetworkAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH, state_size=retecs.DEFAULT_STATE_SIZE,
                                action_size=1,
                                hidden_size=retecs.DEFAULT_NO_HIDDEN_NODES), retecs.preprocess_continuous,
            reward.tcfail),
        lambda: (
            agents.TableauAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH, learning_rate=retecs.DEFAULT_LEARNING_RATE,
                                state_size=retecs.DEFAULT_STATE_SIZE,
                                action_size=retecs.DEFAULT_NO_ACTIONS, epsilon=retecs.DEFAULT_EPSILON),
            retecs.preprocess_discrete, reward.timerank)
    ]

    for time_ratio in scheduling_time_ratios:
        for i, get_agent in enumerate(ags):
            agent, preprocessor, reward_function = get_agent()
            file_appendix = 'rq0_%s_schedtime%d_%d' % (agent.name, time_ratio * 100, iteration)

            scenario = get_scenario('gsdtsr')

            rl_learning = retecs.PrioLearning(agent=agent,
                                              scenario_provider=scenario,
                                              reward_function=reward_function,
                                              preprocess_function=preprocessor,
                                              file_prefix=file_appendix,
                                              dump_interval=100,
                                              validation_interval=0,
                                              output_dir=DATA_DIR)
            res = rl_learning.train(no_scenarios=CI_CYCLES,
                                    print_log=False,
                                    plot_graphs=False,
                                    save_graphs=False,
                                    collect_comparison=i == 0)
            avg_napfd.append(res)

    return avg_napfd


def visualize():
    search_pattern = 'rq0_*_schedtime*_stats.p'
    filename = 'rq0_scheduling_time'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)

    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    # df = df[~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])]

    rel_df = df.groupby(['agent', 'sched_time'], as_index=False).mean()
    rel_df['napfd'] = rel_df['napfd'] / max(rel_df['napfd']) * 100

    for method in method_names.keys():
        rel_df.loc[rel_df['agent'] == method, 'agent'] = method_names[method]

    fig = plt.figure(figsize=figsize_column(1.05, 0.9))
    ax = sns.barplot(x='sched_time', y='napfd', hue='agent', data=rel_df, figure=fig,
                     hue_order=['Network', 'Tableau', 'Sorting', 'Weighting', 'Random'])
    ax.set_xlabel('Scheduling Time Ratio (in \% of $\mathcal{T}_i.duration$)')
    ax.set_ylabel('\% of best result')
    ax.set_xticklabels([int(x * 100) for x in scheduling_time_ratios])
    plt.locator_params(axis='y', nbins=5)

    ax.set_ylim([40, 100])
    ax.legend(title=None, loc=3, frameon=True, ncol=2, bbox_to_anchor=(0., 1.02, 1., .102), mode="expand",
              borderaxespad=0.)
    ax.set_axisbelow(True)
    ax.yaxis.grid(zorder=0)

    fig.tight_layout()
    save_figures(fig, filename)
    plt.clf()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        run_experiments(exp_scheduling_time, parallel=PARALLEL)

    if VISUALIZE_RESULTS:
        visualize()
