#!/usr/bin/env python
from __future__ import division, print_function
import agents
import argparse
import reward
import datetime
import numpy as np
import scenarios
import sys
import time
import os.path
#import plot_stats

try:
    import cPickle as pickle
except:
    import pickle

DEFAULT_NO_SCENARIOS = 1000
DEFAULT_NO_ACTIONS = 100
DEFAULT_HISTORY_LENGTH = 10
DEFAULT_STATE_SIZE = DEFAULT_HISTORY_LENGTH + 1
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPSILON = 0.2
DEFAULT_DUMP_INTERVAL = 100
DEFAULT_VALIDATION_INTERVAL = 100
DEFAULT_PRINT_LOG = False
DEFAULT_PLOT_GRAPHS = False
DEFAULT_NO_HIDDEN_NODES = 12
DEFAULT_TODAY = datetime.datetime.today()


def recency_weighted_avg(values, alpha):
    return sum(np.power(alpha, range(0, len(values))) * values) / len(values)


def preprocess_continuous(state, scenario_metadata, histlen):
    if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
        time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
            scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
    else:
        time_since = 0

    history = [1 if res else 0 for res in state['LastResults'][0:histlen]]

    if len(history) < histlen:
        history.extend([1] * (histlen - len(history)))

    row = [
        state['Duration'] / scenario_metadata['totalTime'],
        time_since
    ]
    row.extend(history)

    return tuple(row)


def preprocess_discrete(state, scenario_metadata, histlen):
    if scenario_metadata['maxDuration'] > scenario_metadata['minDuration']:
        duration = (scenario_metadata['maxDuration'] - state['Duration']) / (
            scenario_metadata['maxDuration'] - scenario_metadata['minDuration'])
    else:
        duration = 0

    if duration > 0.66:
        duration_group = 2
    elif duration > 0.33:
        duration_group = 1
    else:
        duration_group = 0

    if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
        time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
            scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
    else:
        time_since = 0

    if time_since > 0.66:
        time_group = 2
    elif time_since > 0.33:
        time_group = 1
    else:
        time_group = 0

    history = [1 if res else 0 for res in state['LastResults'][0:histlen]]

    if len(history) < histlen:
        history.extend([1] * (histlen - len(history)))

    row = [
        duration_group,
        time_group
    ]
    row.extend(history)

    return tuple(row)


def process_scenario(agent, sc, preprocess):
    scenario_metadata = sc.get_ta_metadata()

    if agent.single_testcases:
        for row in sc.testcases():
            # Build input vector: preprocess the observation
            x = preprocess(row, scenario_metadata, agent.histlen)
            action = agent.get_action(x)
            row['CalcPrio'] = action  # Store prioritization
    else:
        states = [preprocess(row, scenario_metadata, agent.histlen) for row in sc.testcases()]
        actions = agent.get_all_actions(states)

        for (tc_idx, action) in enumerate(actions):
            sc.set_testcase_prio(action, tc_idx)

    # Submit prioritized file for evaluation
    # step the environment and get new measurements
    return sc.submit()


class PrioLearning(object):
    def __init__(self, agent, scenario_provider, file_prefix, reward_function, output_dir, preprocess_function,
                 dump_interval=DEFAULT_DUMP_INTERVAL, validation_interval=DEFAULT_VALIDATION_INTERVAL):
        self.agent = agent
        self.scenario_provider = scenario_provider
        self.reward_function = reward_function
        self.preprocess_function = preprocess_function
        self.replay_memory = agents.ExperienceReplay()
        self.validation_res = []

        self.dump_interval = dump_interval
        self.validation_interval = validation_interval

        self.today = DEFAULT_TODAY

        self.file_prefix = file_prefix
        self.val_file = os.path.join(output_dir, '%s_val' % file_prefix)
        self.stats_file = os.path.join(output_dir, '%s_stats' % file_prefix)
        self.agent_file = os.path.join(output_dir, '%s_agent' % file_prefix)

    def run_validation(self, scenario_count):
        val_res = self.validation()

        for (key, res) in val_res.items():
            res = {
                'scenario': key,
                'step': scenario_count,
                'detected': res[0],
                'missed': res[1],
                'ttf': res[2],
                'apfd': res[3],
                'napfd': res[4],
                'recall': res[5],
                'avg_precision': res[6],
                'order': res[8]
                #etected_failures, undetected_failures, ttf, apfd, napfd, recall, avg_precision, detection_ranks, order
                # res[4] are the detection ranks
            }

            self.validation_res.append(res)

    def validation(self):
        self.agent.train_mode = False
        val_scenarios = self.scenario_provider.get_validation()
        keys = [sc.name for sc in val_scenarios]
        results = [self.process_scenario(sc)[0] for sc in val_scenarios]
        self.agent.train_mode = True
        return dict(zip(keys, results))

    def process_scenario(self, sc):
        result = process_scenario(self.agent, sc, self.preprocess_function)
        reward = self.reward_function(result, sc)
        self.agent.reward(reward)
        return result, reward

    def replay_experience(self, batch_size):
        batch = self.replay_memory.get_batch(batch_size)

        for sc in batch:
            (result, reward) = self.process_scenario(sc)
            print('Replay Experience: %s / %.2f' % (result, np.mean(reward)))

    def train(self, no_scenarios, print_log, plot_graphs, save_graphs, collect_comparison=False):
        stats = {
            'scenarios': [],
            'rewards': [],
            'durations': [],
            'detected': [],
            'missed': [],
            'ttf': [],
            'napfd': [],
            'apfd': [],
            'recall': [],
            'avg_precision': [],
            'result': [],
            'step': [],
            'order': [],
            'env': self.scenario_provider.name,
            'agent': self.agent.name,
            'action_size': self.agent.action_size,
            'history_length': self.agent.histlen,
            'rewardfun': self.reward_function.__name__,
            'sched_time': self.scenario_provider.avail_time_ratio,
            'hidden_size': 'x'.join(str(x) for x in self.agent.hidden_size) if hasattr(self.agent, 'hidden_size') else 0
        }

        if collect_comparison:
            cmp_agents = {
                'heur_sort': agents.HeuristicSortAgent(self.agent.histlen),
                'heur_weight': agents.HeuristicWeightAgent(self.agent.histlen),
                'heur_random': agents.RandomAgent(self.agent.histlen)
            }

            stats['comparison'] = {}

            for key in cmp_agents.keys():
                stats['comparison'][key] = {
                    'detected': [],
                    'missed': [],
                    'ttf': [],
                    'apfd': [],
                    'napfd': [],
                    'recall': [],
                    'avg_precision': [],
                    'durations': []
                }

        sum_actions = 0
        sum_scenarios = 0
        sum_detected = 0
        sum_missed = 0
        sum_reward = 0

        for (i, sc) in enumerate(self.scenario_provider, start=1):
            if i > no_scenarios:
                break

            start = time.time()

            if print_log:
                print('ep %d:\tscenario %s\t' % (sum_scenarios + 1, sc.name), end='')

            (result, reward) = self.process_scenario(sc)

            end = time.time()

            # Statistics
            sum_detected += result[0]
            sum_missed += result[1]
            sum_reward += np.mean(reward)
            sum_actions += 1
            sum_scenarios += 1
            duration = end - start

            stats['scenarios'].append(sc.name)
            stats['rewards'].append(np.mean(reward))
            stats['durations'].append(duration)
            stats['detected'].append(result[0])
            stats['missed'].append(result[1])
            stats['ttf'].append(result[2])
            stats['apfd'].append(result[3])
            stats['napfd'].append(result[4])
            stats['recall'].append(result[5])
            stats['avg_precision'].append(result[6])
            stats['order'].append(result[7])
            stats['result'].append(result)
            stats['step'].append(sum_scenarios)

            if print_log:
                print(' finished, reward: %.2f,\trunning mean: %.4f,\tduration: %.1f,\tapfd: %.2f ,\tnpfd: %.2f ' %
                      (np.mean(reward), sum_reward / sum_scenarios, duration, result[3], result[4]))

            if collect_comparison:
                for key in stats['comparison'].keys():
                    start = time.time()
                    cmp_res = process_scenario(cmp_agents[key], sc, preprocess_discrete)
                    end = time.time()
                    stats['comparison'][key]['detected'].append(cmp_res[0])
                    stats['comparison'][key]['missed'].append(cmp_res[1])
                    stats['comparison'][key]['ttf'].append(cmp_res[2])
                    stats['comparison'][key]['apfd'].append(cmp_res[3])
                    stats['comparison'][key]['napfd'].append(cmp_res[4])
                    stats['comparison'][key]['recall'].append(cmp_res[5])
                    stats['comparison'][key]['avg_precision'].append(cmp_res[6])
                    stats['comparison'][key]['durations'].append(end - start)

            # Data Dumping
            if self.dump_interval > 0 and sum_scenarios % self.dump_interval == 0:
                pickle.dump(stats, open(self.stats_file + '.p', 'wb'))

            if self.validation_interval > 0 and (sum_scenarios == 1 or sum_scenarios % self.validation_interval == 0):
                if print_log:
                    print('ep %d:\tRun test... ' % sum_scenarios, end='')

                self.run_validation(sum_scenarios)
                pickle.dump(self.validation_res, open(self.val_file + '.p', 'wb'))

                if print_log:
                    print('done')

        if self.dump_interval > 0:
            self.agent.save(self.agent_file)
            pickle.dump(stats, open(self.stats_file + '.p', 'wb'))

        if plot_graphs:
            plot_stats.plot_stats_single_figure(self.file_prefix, self.stats_file + '.p', self.val_file + '.p', 1,
                                                plot_graphs=plot_graphs, save_graphs=save_graphs)

        if save_graphs:
            plot_stats.plot_stats_separate_figures(self.file_prefix, self.stats_file + '.p', self.val_file + '.p', 1,
                                                   plot_graphs=False, save_graphs=save_graphs)

        apfds = [i for i in stats['apfd'] if i != 0]
        napfds = [i for i in stats['napfd'] if i != 0]
        print("APFD = " + str(np.mean(apfds)))
        print("NAPFD = " + str(np.mean(napfds)))
        return np.mean(napfds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent',
                        choices=['tableau', 'network', 'heur_random', 'heur_sort', 'heur_weight'], default='network')
    parser.add_argument('-sp', '--scenario-provider',
                        choices=['random', 'incremental', 'paintcontrol', 'iofrol', 'gsdtsr'], default='iofrol')
    parser.add_argument('-r', '--reward', choices=['binary', 'failcount', 'timerank', 'tcfail'], default='failcount')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-hist', '--histlen', type=int, default=DEFAULT_HISTORY_LENGTH)
    parser.add_argument('-eps', '--epsilon', type=float, default=DEFAULT_EPSILON)
    parser.add_argument('-lr', '--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('-as', '--actions', type=int, default=DEFAULT_NO_ACTIONS)
    parser.add_argument('-ns', '--hiddennet', type=int, default=DEFAULT_NO_HIDDEN_NODES)
    parser.add_argument('-n', '--no-scenarios', type=int, default=DEFAULT_NO_SCENARIOS)
    parser.add_argument('-d', '--dump_interval', type=int, default=DEFAULT_DUMP_INTERVAL)
    parser.add_argument('-v', '--validation_interval', type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument('-o', '--output_dir', default='.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False)
    parser.add_argument('--plot-graphs', action='store_true', default=False)
    parser.add_argument('--save-graphs', action='store_true', default=False)
    parser.add_argument('--comparable', action='store_true', default=False)
    args = parser.parse_args()

    state_size = 2 + args.histlen
    preprocess_function = preprocess_discrete

    if args.agent == 'tableau':
        agent = agents.TableauAgent(learning_rate=args.learning_rate, state_size=state_size, action_size=args.actions,
                                    epsilon=args.epsilon, histlen=args.histlen)
    elif args.agent == 'network':
        if args.reward in ('binary', 'tcfail'):
            action_size = 1
        else:
            action_size = 2

        agent = agents.NetworkAgent(state_size=state_size, action_size=action_size, hidden_size=args.hiddennet,
                                    histlen=args.histlen)
    elif args.agent == 'heur_random':
        agent = agents.RandomAgent(histlen=args.histlen)
    elif args.agent == 'heur_sort':
        agent = agents.HeuristicSortAgent(histlen=args.histlen)
    elif args.agent == 'heur_weight':
        agent = agents.HeuristicWeightAgent(histlen=args.histlen)
    else:
        print('Unknown Agent')
        sys.exit()

    if args.scenario_provider == 'random':
        scenario_provider = scenarios.RandomScenarioProvider()
    elif args.scenario_provider == 'incremental':
        scenario_provider = scenarios.IncrementalScenarioProvider(episode_length=args.no_scenarios)
    elif args.scenario_provider == 'paintcontrol':
        scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/paintcontrol.csv')
        #scenario_provider = scenarios.FileBasedSubsetScenarioProvider(scheduleperiod=datetime.timedelta(days=1),
        #                                                              tcfile='DATA/tc_data_paintcontrol.csv',
        #                                                              solfile='DATA/tc_sol_paintcontrol.csv')
        args.validation_interval = 0
    elif args.scenario_provider == 'iofrol':
        scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/iofrol.csv')
        args.validation_interval = 0
    elif args.scenario_provider == 'gsdtsr':
        scenario_provider = scenarios.IndustrialDatasetScenarioProvider(tcfile='DATA/gsdtsr.csv')
        args.validation_interval = 0

    if args.reward == 'binary':
        reward_function = reward.binary_positive_detection_reward
    elif args.reward == 'failcount':
        reward_function = reward.failcount
    elif args.reward == 'timerank':
        reward_function = reward.timerank
    elif args.reward == 'tcfail':
        reward_function = reward.tcfail

    prefix = '{}_{}_{}_lr{}_as{}_n{}_eps{}_hist{}_{}'.format(args.agent, args.scenario_provider, args.reward,
                                                             args.learning_rate, args.actions, args.no_scenarios,
                                                             args.epsilon, args.histlen, args.prefix)

    rl_learning = PrioLearning(agent=agent,
                               scenario_provider=scenario_provider,
                               reward_function=reward_function,
                               preprocess_function=preprocess_function,
                               file_prefix=prefix,
                               dump_interval=args.dump_interval,
                               validation_interval=args.validation_interval,
                               output_dir=args.output_dir)
    avg_napfd = rl_learning.train(no_scenarios=args.no_scenarios, print_log=not args.quiet,
                                  plot_graphs=args.plot_graphs,
                                  save_graphs=args.save_graphs,
                                  collect_comparison=args.comparable)
    print(avg_napfd)
