from typing import Any, Union

import numpy as np
import gym
from gym import spaces
from sklearn import preprocessing
from ci_cycle import CICycleLog
from Config import Config


class CIPairWiseEnv(gym.Env):
    def __init__(self, cycle_logs: CICycleLog, conf: Config):
        super(CIPairWiseEnv, self).__init__()
        self.conf = conf
        self.reward_range = (-1, 1)
        self.cycle_logs = cycle_logs
        self.initial_observation = cycle_logs.test_cases.copy()
        self.test_cases_vector = self.initial_observation.copy()
        self.current_indexes = [0, 1]
        self.sorted_test_cases_vector = []
        self.current_obs = np.zeros((2, self.conf.win_size + 2))
        self.current_obs = self.get_pair_data(self.current_indexes)

        # self.number_of_actions = len(self.cycle_logs.test_cases)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, self.conf.win_size + 2))  # ID, execution time and LastResults

    def get_test_cases_vector(self):
        return self.test_cases_vector

    def get_pair_data(self, current_indexes):
        i = 0
        temp_obs = np.zeros((2, self.conf.win_size + 2))
        for test_index in current_indexes:
            temp_obs[i, :] = self.cycle_logs.export_test_case(self.test_cases_vector[test_index],
                                                           "list_avg_exec_with_failed_history",
                                                           self.conf.padding_digit,
                                                           self.conf.win_size)
            i = i + 1
        temp_obs = preprocessing.normalize(temp_obs, axis=0, norm='max')
        return temp_obs

    def render(self, mode='human'):
        pass

    def reset(self):
        self.test_cases_vector = self.initial_observation.copy()
        self.current_indexes = [0, 1]
        self.current_obs = self.get_pair_data(self.current_indexes)
        return self.current_obs

    def _next_observation(self, index):
        self.current_obs = self.get_pair_data(self.current_indexes)
        return self.current_obs

    def _initial_obs(self):
        return self.initial_observation

    ## the reward function must be called before updating the observation
    def _calculate_reward(self, test_case_index):
        if test_case_index == 0:
            selected_test_case = self.test_cases_vector[self.current_indexes[0]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[1]]
        else:
            selected_test_case = self.test_cases_vector[self.current_indexes[1]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[0]]
        if selected_test_case['verdict'] > no_selected_test_case['verdict']:
            reward = 1
        elif selected_test_case['verdict'] < no_selected_test_case['verdict']:
            reward = 0
        elif selected_test_case['avg_exec_time'] <= no_selected_test_case['avg_exec_time']:
            reward = 0.5
        elif selected_test_case['avg_exec_time'] > no_selected_test_case['avg_exec_time']:
            reward = 0.1
        return reward

    def swapPositions(self, l, pos1, pos2):
        l[pos1], l[pos2] = l[pos2], l[pos1]
        return l

    def step(self, test_case_index):
        done = False
        reward = self._calculate_reward(test_case_index)
        if test_case_index == 1:
            self.swapPositions(self.test_cases_vector, self.current_indexes[0], self.current_indexes[1])
        if self.current_indexes[1] < (len(self.test_cases_vector) - 1):
            self.current_indexes[1] = self.current_indexes[1] + 1
        elif (self.current_indexes[1] == len(self.test_cases_vector) - 1) and \
                (self.current_indexes[0] < len(self.test_cases_vector) - 2):
            self.current_indexes[0] = self.current_indexes[0] + 1
            self.current_indexes[1] = self.current_indexes[0] + 1
        else:
            done = True
            ## a2c reset the env when the epsiode is done, so we need to copy the result of test cases
            self.sorted_test_cases_vector = self.test_cases_vector.copy()

        self.current_obs = self._next_observation(test_case_index)
        return self.current_obs, reward, done, {}
