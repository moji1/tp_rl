#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ1

from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = True
RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True


def visualize():
    search_pattern = 'rq_*_stats.p'
    filename = 'rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)
    df = stats.load_stats_dataframe(iteration_results, aggregated_results)

    pure_df = df[(~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])) & (df['detected'] + df['missed']) > 0]
    mean_df = pure_df.groupby(['step', 'env', 'agent', 'rewardfun'], as_index=False).mean()

    # One subplot per data set (= one row in the paper)
    # for env in mean_df['env'].unique():
    #     plotname = 'rq1_napfd_%s' % env
    #     fig, axarr = plt.subplots(1, 3, sharey=True, figsize=figsize_text(1.0, 0.45))
    #     i = 0
    #
    #     for rewardfun in mean_df['rewardfun'].unique():
    #         for agidx, (labeltext, agent, linestyle) in enumerate(
    #                 [('Network', 'mlpclassifier', '-'), ('Tableau', 'tableau', '--')]):
    #             rel_df = mean_df[(mean_df['env'] == env) & (mean_df['rewardfun'] == rewardfun)]
    #             rel_df[rel_df['agent'] == agent].plot(x='step', y='napfd', label=labeltext, ylim=[0, 1], linewidth=0.8,
    #                                                   style=linestyle, color=sns.color_palette()[agidx], ax=axarr[i])
    #
    #             x = rel_df.loc[rel_df['agent'] == agent, 'step']
    #             y = rel_df.loc[rel_df['agent'] == agent, 'napfd']
    #             trend = np.poly1d(np.polyfit(x, y, 1))
    #             axarr[i].plot(x, trend(x), linestyle, color='k', linewidth=0.8)
    #
    #         axarr[i].set_xlabel('CI Cycle')
    #         axarr[i].legend_.remove()
    #         axarr[i].set_title(reward_names[rewardfun])
    #         axarr[i].set_xticks(np.arange(0, 350, 30), minor=False)
    #         axarr[i].set_xticklabels([0, '', 60, '', 120, '', 180, '', 240, '', 300], minor=False)
    #
    #         axarr[i].xaxis.grid(True, which='minor')
    #
    #         if i == 0:
    #             axarr[i].set_ylabel('NAPFD')
    #             axarr[i].legend(loc=2, frameon=True)
    #
    #         i += 1
    #
    #     fig.tight_layout()
    #     fig.subplots_adjust(wspace=0.08)
    #     save_figures(fig, plotname)
    #     plt.clf()

    # One groupplot
    fig, axarr = plt.subplots(3, 3, sharey=True, sharex=True, figsize=figsize_text(1.0, 1.2))
    plotname = 'rq1_napfd'
    subplot_labels = ['(a)', '(b)', '(c)']

    for column, env in enumerate(sorted(mean_df['env'].unique(), reverse=True)):
        for row, rewardfun in enumerate(mean_df['rewardfun'].unique()):
            for agidx, (labeltext, agent, linestyle) in enumerate(
                    [('Network', 'mlpclassifier', '-'), ('Tableau', 'tableau', '--')]):
                rel_df = mean_df[(mean_df['env'] == env) & (mean_df['rewardfun'] == rewardfun)]
                rel_df[rel_df['agent'] == agent].plot(x='step', y='napfd', label=labeltext, ylim=[0, 1], linewidth=0.8,
                                                      style=linestyle, color=sns.color_palette()[agidx], ax=axarr[row, column])

                x = rel_df.loc[rel_df['agent'] == agent, 'step']
                y = rel_df.loc[rel_df['agent'] == agent, 'napfd']
                trend = np.poly1d(np.polyfit(x, y, 1))
                axarr[row, column].plot(x, trend(x), linestyle, color='k', linewidth=0.8)

            axarr[row, column].legend_.remove()

            axarr[row, column].set_xticks(np.arange(0, 350, 30), minor=False)
            axarr[row, column].set_xticklabels([0, '', 60, '', 120, '', 180, '', 240, '', 300], minor=False)
            axarr[row, column].xaxis.grid(True, which='major')

            if column == 1:
                axarr[row, column].set_title('\\textbf{%s %s}' % (subplot_labels[row], reward_names[rewardfun]))

            if row == 0:
                if column == 1:
                    axarr[row, column].set_title('%s\n\\textbf{%s %s}' % (env_names[env], subplot_labels[row], reward_names[rewardfun]))
                else:
                    axarr[row, column].set_title(env_names[env] + '\n')
            elif row == 2:
                axarr[row, column].set_xlabel('CI Cycle')

            if column == 0:
                axarr[row, column].set_ylabel('NAPFD')

            if row == 0 and column == 0:
                axarr[row, column].legend(loc=2, ncol=2, frameon=True, bbox_to_anchor=(0.065, 1.1))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06, hspace=0.3)
    save_figures(fig, plotname)
    plt.clf()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        results=run_experiments(exp_run_industrial_datasets, parallel=True)

    if VISUALIZE_RESULTS:
        visualize()
