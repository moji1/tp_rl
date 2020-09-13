from typing import Any, Union

import numpy as np
import gym
from gym import spaces
from Config import Config
import random

from ci_cycle import CICycleLog


class CIListWiseEnvMultiAction(gym.Env):
    def __init__(self, cycle_logs: CICycleLog, conf: Config):
        super(CIListWiseEnvMultiAction, self).__init__()
        self.reward_range = (-1, 1)
        self.cycle_logs = cycle_logs
        self.padding_value = -1
        self.conf = conf
        random.shuffle(self.cycle_logs.test_cases)
        self.optimal_order = cycle_logs.get_optimal_order()
        self.testcase_vector_size = self.cycle_logs.get_test_case_vector_length(cycle_logs.test_cases[0],
                                                                                self.conf.win_size)
        self.current_obs = self.cycle_logs.export_test_cases("list_avg_exec_with_failed_history", -1,
                                                             self.conf.max_test_cases_count, self.conf.win_size,
                                                             self.testcase_vector_size)
        self.initial_observation = np.copy(self.current_obs)
        # self.number_of_actions = len(self.cycle_logs.test_cases)
        #self.action_space = spaces.Discrete(conf.max_test_cases_count)
        action_spec = [conf.max_test_cases_count] * conf.max_test_cases_count
        self.action_space = spaces.MultiDiscrete(action_spec)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.current_obs.shape[0],
                                                   self.current_obs.shape[1]))  # ID, execution time and LastResults
        self.sorted_test_cases = []
        # self.APFD = 0
        # self.ID = 0
        # self.fail_rank = []
        # self.current_obs = self.cycle_logs.export_test_cases("list_avg_exec_with_failed_history", 0.001,
        #                                                     max_test_cases_count, win_size, 0)

    def render(self, mode='human'):
        pass

    def reset(self):
        random.shuffle(self.cycle_logs.test_cases)
        self.current_obs = self.cycle_logs.export_test_cases("list_avg_exec_with_failed_history", -1,
                                                             self.conf.max_test_cases_count, self.conf.win_size,
                                                             self.testcase_vector_size)
        self.initial_observation = np.copy(self.current_obs)
        return self.current_obs

    def _next_observation(self, index):
        return self.initial_observation

    def _initial_obs(self):
        return self.initial_observation

    ## the reward function must be called before updating the observation
    def _calculate_reward(self, sorted_test_cases):
        if self.cycle_logs.get_failed_test_cases_count() > 0:
            apfd = self.cycle_logs.calc_APFD_ordered_vector(sorted_test_cases)
            apfd_optimal = self.cycle_logs.calc_optimal_APFD()
            reward = apfd / apfd_optimal
        else:
            nrpa = self.cycle_logs.calc_NRPA_vector(sorted_test_cases)
            reward = nrpa
        return reward

    def step(self, ranks):
        done = True
        index = 0
        test_case_ranks = {}
        self.sorted_test_cases = []
        for rank in ranks:
            if not (rank in test_case_ranks.keys()):
                test_case_ranks[rank] = []
            test_case_ranks[rank].append(index)
            index += 1
        for i in range(0, self.conf.max_test_cases_count):
            if i in test_case_ranks.keys():
                for j in test_case_ranks[i]:
                    if j < self.cycle_logs.get_test_cases_count():
                        self.sorted_test_cases.append(self.cycle_logs.test_cases[j])

        reward = self._calculate_reward(self.sorted_test_cases)
        self.current_obs = self.reset()
        return self.current_obs, reward, done, {}
