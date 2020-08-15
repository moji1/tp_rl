#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ0 - Number of actions in tableau representation
from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = True
RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True

action_sizes = [2, 5, 10, 25, 50, 75, 100, 150, 200]
sns.set_palette(sns.color_palette("Set1", n_colors=len(action_sizes), desat=.5))


def exp_tableau_action_size(iteration):
    reward_function = reward.timerank

    avg_napfd = []

    for actions in action_sizes:
        for sc in ['paintcontrol', 'iofrol']:
            agent = agents.TableauAgent(learning_rate=retecs.DEFAULT_LEARNING_RATE,
                                        state_size=retecs.DEFAULT_STATE_SIZE,
                                        action_size=actions, epsilon=retecs.DEFAULT_EPSILON,
                                        histlen=retecs.DEFAULT_HISTORY_LENGTH)

            scenario = get_scenario(sc)

            file_appendix = 'rq0_%s_sc%s_as%d_%d' % (agent.name, sc, actions, iteration)

            rl_learning = retecs.PrioLearning(agent=agent,
                                              scenario_provider=scenario,
                                              reward_function=reward_function,
                                              preprocess_function=retecs.preprocess_discrete,
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
    search_pattern = 'rq0_*_as*_stats.p'
    filename = 'rq0_tableau_action_size'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)

    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    df = df[~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])]

    for (i, env) in enumerate(df.env.unique()):
        fig_filename = filename + '_' + env
        rel_df = df[df['env'] == env].groupby(['agent', 'action_size'], as_index=False).mean()
        rel_df['napfd'] = rel_df['napfd'] / max(rel_df['napfd']) * 100

        ax = sns.barplot(x='action_size', y='napfd', data=rel_df, errwidth=0, linewidth=1)
        ax.set_xlabel('No. of Actions')
        ax.set_ylabel('\% of best result')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_xticklabels(action_sizes)
        save_figures(ax.figure, fig_filename)
        plt.clf()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        run_experiments(exp_tableau_action_size, parallel=PARALLEL)

    if VISUALIZE_RESULTS:
        visualize()
