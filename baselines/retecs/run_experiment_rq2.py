#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ2
import glob
import os

from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = True
RUN_EXPERIMENT = False
VISUALIZE_RESULTS = True


def visualize():
    search_pattern = 'rq_*_stats.p'
    filename = 'rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)
    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    df.loc[pd.isnull(df['rewardfun']), 'rewardfun'] = 'none'
    mean_df = df[(df['detected'] + df['missed']) > 0].groupby(['step', 'env', 'agent', 'rewardfun'],
                                                              as_index=False).mean()

    # Single figures
    #for (agent, rewfun) in [('mlpclassifier', 'tcfail'), ('tableau', 'timerank')]:
    # for (agent, rewfun) in [('mlpclassifier', 'tcfail')]:
    #     i = 0
    #
    #     for env in sorted(mean_df['env'].unique(), reverse=True):
    #         plotname = 'rq2_napfd_abs_%s_%s' % (env, agent)
    #         print env
    #
    #         reddf = mean_df[(mean_df['env'] == env) & (
    #         mean_df['agent'].isin([agent, 'heur_sort', 'heur_weight', 'heur_random'])) & (
    #                         mean_df['rewardfun'].isin([rewfun, 'none']))].groupby(['step', 'agent']).mean()[
    #             'napfd'].unstack()
    #         reddf['Heuristic Sort'] = reddf['heur_sort'] - reddf[agent]
    #         reddf['Heuristic Weight'] = reddf['heur_weight'] - reddf[agent]
    #         reddf['Random'] = reddf['heur_random'] - reddf[agent]
    #         del reddf[agent]
    #         del reddf['heur_sort']
    #         del reddf['heur_weight']
    #         del reddf['heur_random']
    #         window_size = 30
    #         r = reddf.groupby(reddf.index // window_size).mean()
    #         xdf = r.stack()
    #         xdf = xdf.reset_index(level=xdf.index.names)
    #         xdf['step'] *= window_size
    #
    #         fig = plt.figure(figsize=(8, 4))
    #         ax = sns.barplot(data=xdf, x='step', y=0, hue='agent', figure=fig)
    #
    #         ax.set_ylabel('')
    #         ax.set_xlabel('CI Cycle')
    #         ax.legend_.remove()
    #
    #         if i == 0:
    #             ax.set_ylabel('NAPFD Difference')
    #             # ax.set_ylim([-0.6, 0.6])
    #             ax.set_title('ABB Paint Control')
    #         elif i == 1:
    #             ax.set_title('ABB IOF/ROL')
    #             ax.legend(ncol=1, loc=1)
    #         elif i == 2 and len(mean_df['env'].unique()) == 3:
    #             ax.set_title('Google GSDTSR')
    #
    #         #ax.set_ylim([-0.65, 0.65])
    #         fig.tight_layout()
    #         save_figures(fig, plotname)
    #         plt.clf()
    #
    #         i += 1


    # Grouped figure
    plotname = 'rq2_napfd_bar_abs_grouped'
    fig, axarr = plt.subplots(1, len(mean_df['env'].unique()), sharey=True, figsize=figsize_text(1.0, 0.5))

    i = 0

    group_df = df[((df['detected'] + df['missed']) > 0) &
                  (df['agent'].isin(['mlpclassifier', 'heur_sort', 'heur_weight', 'heur_random'])) &
                  (df['rewardfun'].isin(['tcfail', 'none']))].groupby(['step', 'env', 'agent', 'iteration'],
                                                                      as_index=False).mean()

    for env in sorted(group_df['env'].unique(), reverse=True):
        ax = axarr[i]

        reddf = group_df[(group_df['env'] == env)].groupby(['step', 'iteration', 'agent']).mean()['napfd'].unstack()
        reddf[method_names['heur_sort']] = reddf['heur_sort'] - reddf['mlpclassifier']
        reddf[method_names['heur_weight']] = reddf['heur_weight'] - reddf['mlpclassifier']
        reddf[method_names['heur_random']] = reddf['heur_random'] - reddf['mlpclassifier']
        del reddf['mlpclassifier']
        del reddf['heur_sort']
        del reddf['heur_weight']
        del reddf['heur_random']

        window_size = 30

        wdf = reddf.stack().reset_index()
        wdf = wdf.set_index(['step'])
        r = wdf.groupby([wdf.index // window_size, 'agent', 'iteration']).mean()
        xdf = r.stack()
        xdf = xdf.reset_index(level=xdf.index.names)
        xdf['step'] *= window_size
        sns.barplot(data=xdf, x='step', y=0, hue='agent', hue_order=[method_names['heur_sort'], method_names['heur_weight'], method_names['heur_random']], ax=ax, linewidth=0.5, palette=sns.color_palette()[2:])

        xl = ax.get_xlim()
        ax.plot(xl, [0, 0], color='k', linestyle='-', zorder=0.5)
        ax.set_xlim(xl)
        ax.set_ylabel('')
        ax.set_xlabel('CI Cycle')
        ax.legend_.remove()
        ax.set_xticklabels([], minor=False)
        ax.set_xticklabels(['', 60, '', 120, '', 180, '', 240, '', 300], minor=True)
        ax.set_xticks(np.arange(0.5, max(wdf.index // window_size)+1), minor=True)
        ax.xaxis.grid(True, which='minor')

        if i == 0:
            ax.set_ylabel('NAPFD Difference')
            #ax.set_ylim([-0.6, 0.6])
            ax.set_title('ABB Paint Control')
            ax.legend(ncol=1, loc=1, frameon=True)
        elif i == 1:
            ax.set_title('ABB IOF/ROL')
        elif i == 2 and len(mean_df['env'].unique()) == 3:
            ax.set_title('Google GSDTSR')

        i += 1

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08)
    save_figures(fig, plotname)
    plt.clf()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        run_experiments(exp_run_industrial_datasets, PARALLEL)

    if VISUALIZE_RESULTS:
        visualize()
