#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ0 - Comparison of reward functions on single data set
from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = True
RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True


def exp_tableau_action_size(iteration):
    reward_funs = {
        'failcount': reward.failcount,
        'timerank': reward.timerank,
        'tcfail': reward.tcfail
    }

    ags = [
        lambda: (
            agents.TableauAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH, learning_rate=retecs.DEFAULT_LEARNING_RATE,
                                state_size=retecs.DEFAULT_STATE_SIZE,
                                action_size=retecs.DEFAULT_NO_ACTIONS, epsilon=retecs.DEFAULT_EPSILON),
            retecs.preprocess_discrete),
        lambda: (agents.NetworkAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH, state_size=retecs.DEFAULT_STATE_SIZE,
                                     action_size=1,
                                     hidden_size=retecs.DEFAULT_NO_HIDDEN_NODES), retecs.preprocess_continuous)
    ]

    avg_napfd = []

    for (reward_name, reward_fun) in reward_funs.items():
        for get_agent in ags:
            agent, preprocessor = get_agent()
            file_appendix = 'rq0_%s_rw%s_%d' % (agent.name, reward_name, iteration)

            scenario = get_scenario('paintcontrol')

            rl_learning = retecs.PrioLearning(agent=agent,
                                              scenario_provider=scenario,
                                              reward_function=reward_fun,
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
    search_pattern = 'rq0_*_rw*_stats.p'
    filename = 'rq0_reward_functions'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)

    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    df = df[~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])]

    rel_df = df.groupby(['agent', 'rewardfun'], as_index=False).mean()
    rel_df['napfd'] = rel_df['napfd'] / max(rel_df['napfd']) * 100

    rel_df.loc[rel_df['agent'] == 'mlpclassifier', 'agent'] = 'Network'
    rel_df.loc[rel_df['agent'] == 'tableau', 'agent'] = 'Tableau'

    rel_df.loc[rel_df['rewardfun'] == 'failcount', 'rewardfun'] = 'Failure Count'
    rel_df.loc[rel_df['rewardfun'] == 'tcfail', 'rewardfun'] = 'Test Case Failure'
    rel_df.loc[rel_df['rewardfun'] == 'timerank', 'rewardfun'] = 'Time-ranked'

    ax = sns.barplot(x='rewardfun', y='napfd', hue='agent', data=rel_df, errwidth=0, linewidth=1)
    ax.set_xlabel('Reward Function')
    ax.set_ylabel('% of best result')
    ax.legend(title=None)

    save_figures(ax.figure, filename)
    plt.clf()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        run_experiments(exp_tableau_action_size, parallel=PARALLEL)

    if VISUALIZE_RESULTS:
        visualize()
