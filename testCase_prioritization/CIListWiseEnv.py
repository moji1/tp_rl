from typing import Any, Union

import numpy as np
import gym
from gym import spaces
from Config import Config

from ci_cycle import CICycleLog


class CIListWiseEnv(gym.Env):
    def __init__(self, cycle_logs: CICycleLog, conf: Config):
        super(CIListWiseEnv, self).__init__()
        self.reward_range = (-1, 1)
        self.cycle_logs = cycle_logs
        self.padding_value = -1
        self.conf = conf
        self.optimal_order= cycle_logs.get_optimal_order()
        self.testcase_vector_size = self.cycle_logs.get_test_case_vector_length(cycle_logs.test_cases[0],
                                                                                self.conf.win_size)
        self.current_obs = self.cycle_logs.export_test_cases("list_avg_exec_with_failed_history", -1,
                                                             self.conf.max_test_cases_count, self.conf.win_size,
                                                             self.testcase_vector_size)
        self.initial_observation = np.copy(self.current_obs)
        # self.number_of_actions = len(self.cycle_logs.test_cases)
        self.action_space = spaces.Discrete(conf.max_test_cases_count)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.current_obs.shape[0],
                                                   self.current_obs.shape[1]))  # ID, execution time and LastResults
        self.agent_results = []
        # self.APFD = 0
        # self.ID = 0
        # self.fail_rank = []
        # self.current_obs = self.cycle_logs.export_test_cases("list_avg_exec_with_failed_history", 0.001,
        #                                                     max_test_cases_count, win_size, 0)

    def render(self, mode='human'):
        pass

    def reset(self):
        self.current_obs = np.copy(self._initial_obs())
        self.reset_agent_results()
        return self.current_obs

    def get_agent_actions(self):
        return self.agent_results

    def reset_agent_results(self):
        self.agent_results = []

    def _next_observation(self, index):
        if self.agent_results.count(index) == 0 and \
                index < self.cycle_logs.get_test_cases_count():
            self.agent_results.append(index)
            self.current_obs[index] = np.repeat(self.padding_value, self.current_obs.shape[1])
        # np.zeros(self.max_size+2)
        return self.current_obs

    def _initial_obs(self):
        return self.initial_observation

    ## the reward function must be called before updating the observation
    def _calculate_reward(self, test_case_index):
        if test_case_index >= self.cycle_logs.get_test_cases_count() or \
                (np.repeat(self.padding_value, self.current_obs.shape[1])
                 == self.current_obs[test_case_index]).all():
            return 0
        assigned_rank = len(set(self.agent_results))
        optimal_rank = self.optimal_order.index(self.cycle_logs.test_cases[test_case_index])
        normalized_optimal_rank = optimal_rank/self.cycle_logs.get_test_cases_count()
        normalized_assigned_rank = assigned_rank / self.cycle_logs.get_test_cases_count()
        reward = 1 - (normalized_assigned_rank-normalized_optimal_rank)**2
        return reward

    def _calculate_reward1(self, test_case_index):
        if test_case_index >= self.cycle_logs.get_test_cases_count() or \
                (np.repeat(self.padding_value, self.current_obs.shape[1])
                 == self.current_obs[test_case_index]).all():
            return -1  ## make sure that the agent (1) does not take repeated actions and
            # (2) does not select dummy test cases that are added to make the action space are unified
        rank = len(self.agent_results) + 1
        if (self.cycle_logs.get_test_cases_count() - 1)>0:
            norm_rank = (rank - 1) / (self.cycle_logs.get_test_cases_count() - 1)
        else:
            norm_rank = 0
        norm_exec_time = self.cycle_logs.get_test_case_last_exec_time_normalized(test_case_index)
        verdict = self.cycle_logs.get_test_case_verdict(test_case_index)
        # self.current_obs[action] [1]
        reward = verdict - abs(norm_rank - norm_exec_time)
        return reward

    def step(self, test_case_index):
        done = False
        reward = self._calculate_reward(test_case_index)
        self.current_obs = self._next_observation(test_case_index)
        if len(set(self.agent_results)) == self.cycle_logs.get_test_cases_count():
            done = True

        return self.current_obs, reward, done, {}
