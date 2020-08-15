import datetime
import multiprocessing
import numpy as np
import pandas as pd
import glob
import os
import matplotlib as mpl

USE_LATEX = False

if USE_LATEX:
    mpl.use('pgf')
else:
    mpl.use('Agg')


def figsize_column(scale, height_ratio=1.0):
    fig_width_pt = 240  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def figsize_text(scale, height_ratio=1.0):
    fig_width_pt = 504  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.figsize": figsize_column(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}

import matplotlib.pyplot as plt
import seaborn as sns

if USE_LATEX:
    sns.set_style('whitegrid', pgf_with_latex)
else:
    sns.set_style('whitegrid')

sns.set_context('paper')
sns.set_palette(sns.color_palette("Set1", n_colors=8, desat=.5))

from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import agents
import reward
import retecs
import scenarios
import stats

ITERATIONS = 30
CI_CYCLES = 1000

DATA_DIR = 'RESULTS'
FIGURE_DIR = 'RESULTS'
PARALLEL = True
PARALLEL_POOL_SIZE = 2

RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True

method_names = {
    'mlpclassifier': 'Network',
    'tableau': 'Tableau',
    'heur_random': 'Random',
    'heur_sort': 'Sorting',
    'heur_weight': 'Weighting'
}

reward_names = {
    'failcount': 'Failure Count Reward',
    'tcfail': 'Test Case Failure Reward',
    'timerank': 'Time-ranked Reward'
}

env_names = {
    'paintcontrol': 'ABB Paint Control',
    'iofrol': 'ABB IOF/ROL',
     'gsdtsr': 'GSDTSR'
}


def get_scenario(name):
    if name == 'incremental':
        sc = scenarios.IncrementalScenarioProvider(episode_length=CI_CYCLES)
    elif name == 'paintcontrol':
        sc = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/paintcontrol.csv')
    elif name == 'iofrol':
        sc = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/iofrol.csv')
    elif name == 'gsdtsr':
        sc = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/gsdtsr.csv')

    return sc


def run_experiments(exp_fun, parallel=PARALLEL):
    if parallel:
        p = multiprocessing.Pool(PARALLEL_POOL_SIZE)
        avg_res = p.map(exp_fun, range(ITERATIONS))
    else:
        avg_res = [exp_fun(i) for i in range(ITERATIONS)]

    print('Ran experiments: %d results' % len(avg_res))


def exp_run_industrial_datasets(iteration, datasets=['paintcontrol', 'iofrol', 'gsdtsr']):
    ags = [
        lambda: (
            agents.TableauAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH, learning_rate=retecs.DEFAULT_LEARNING_RATE,
                                state_size=retecs.DEFAULT_STATE_SIZE,
                                action_size=retecs.DEFAULT_NO_ACTIONS, epsilon=retecs.DEFAULT_EPSILON),
            retecs.preprocess_discrete, reward.timerank),
        lambda: (agents.NetworkAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH, state_size=retecs.DEFAULT_STATE_SIZE,
                                     action_size=1,
                                     hidden_size=retecs.DEFAULT_NO_HIDDEN_NODES), retecs.preprocess_continuous,
                 reward.tcfail)
    ]

    reward_funs = {
        'failcount': reward.failcount,
        'timerank': reward.timerank,
        'tcfail': reward.tcfail
    }

    avg_napfd = []

    for i, get_agent in enumerate(ags):
        for sc in datasets:
            for (reward_name, reward_fun) in reward_funs.items():
                agent, preprocessor, _ = get_agent()
                file_appendix = 'rq_%s_%s_%s_%d' % (agent.name, sc, reward_name, iteration)

                scenario = get_scenario(sc)

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
                                        collect_comparison=(i == 0))
                avg_napfd.append(res)
                print(agent.name+","+sc+","+reward_name+","+str(res))

    return avg_napfd


def save_figures(fig, filename):
    if USE_LATEX:
        fig.savefig(os.path.join(FIGURE_DIR, filename + '.pgf'), bbox_inches='tight')

    fig.savefig(os.path.join(FIGURE_DIR, filename + '.pdf'), bbox_inches='tight')
