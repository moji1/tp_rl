#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ0 - Different history lengths
import glob
import os
from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = True
RUN_EXPERIMENT = True
EVALUATE = True
VISUALIZE_RESULTS = False
history_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 50]


def exp_history_length(iteration):
    avg_napfd = []

    ags = [
        lambda hl: (agents.TableauAgent(histlen=hl, learning_rate=retecs.DEFAULT_LEARNING_RATE,
                                        state_size=retecs.DEFAULT_STATE_SIZE,
                                        action_size=retecs.DEFAULT_NO_ACTIONS, epsilon=retecs.DEFAULT_EPSILON),
                    retecs.preprocess_discrete, reward.timerank),
        lambda hl: (agents.NetworkAgent(histlen=hl, state_size=retecs.DEFAULT_STATE_SIZE, action_size=1,
                                        hidden_size=retecs.DEFAULT_NO_HIDDEN_NODES), retecs.preprocess_continuous,
                    reward.tcfail),
    ]

    for histlen in history_lengths:
        for get_agent in ags:
            agent, preprocessor, reward_function = get_agent(histlen)
            file_appendix = 'rq0_%s_histlen%d_%d' % (agent.name, histlen, iteration)

            scenario = get_scenario('paintcontrol')

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
                                    collect_comparison=False)
            avg_napfd.append(res)

    return avg_napfd


def visualize():
    search_pattern = 'rq0_*_histlen*_stats.p'
    filename = 'rq0_history_length'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)

    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    df = df[~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])]

    rel_df = df.groupby(['agent', 'history_length'], as_index=False).mean()
    rel_df['napfd'] = rel_df['napfd'] / max(rel_df['napfd']) * 100

    rel_df.loc[rel_df['agent'] == 'mlpclassifier', 'agent'] = method_names['mlpclassifier']
    rel_df.loc[rel_df['agent'] == 'tableau', 'agent'] = method_names['tableau']

    fig = plt.figure(figsize=figsize_column(1.0))
    ax = sns.barplot(x='history_length', y='napfd', hue='agent', data=rel_df, figure=fig)
    ax.set_xlabel('History Length')
    ax.set_ylabel('\% of best result')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xticklabels(history_lengths)
    ax.set_ylim([60, 100])
    plt.locator_params(axis='y', nbins=5)

    #    state_space = [3 * 3 * (2 ** hl) for hl in history_lengths]
    #    ax2 = ax.twinx()
    #    ax2.semilogy(range(len(history_lengths)), state_space, color='k', linestyle='--')
    #    ax2.set_ylabel('State Space Size')
    #    ax2.tick_params('y')

    ax.legend(title=None, loc=4, frameon=True)

    ax.set_axisbelow(True)
    ax.yaxis.grid(zorder=0)
    #   ax2.set_axisbelow(True)
    #   ax2.yaxis.grid(zorder=0)

    fig.tight_layout()
    save_figures(fig, filename)
    plt.clf()



if __name__ == '__main__':
    if RUN_EXPERIMENT:
        results=run_experiments(exp_history_length, parallel=PARALLEL)

    if VISUALIZE_RESULTS:
        visualize()
