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
        self.testcase_vector_size=self.cycle_logs.get_test_case_vector_length(cycle_logs.test_cases[0],self.conf.win_size)
        self.initial_observation = cycle_logs.test_cases.copy()
        self.test_cases_vector = self.initial_observation.copy()
        self.test_cases_vector_temp=[]
        self.current_indexes = [0, 1]
        self.sorted_test_cases_vector = []
        self.current_obs = np.zeros((2, self.testcase_vector_size))
        self.width = 1
        self.right = 1
        self.left = 0
        self.end = 2
        self.index = 0
        self.current_indexes[0] = self.index
        self.current_indexes[1] = self.index + self.width
        self.current_obs = self.get_pair_data(self.current_indexes)

        # self.number_of_actions = len(self.cycle_logs.test_cases)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, self.testcase_vector_size))  # ID, execution time and LastResults

    def get_test_cases_vector(self):
        return self.test_cases_vector

    def get_pair_data(self, current_indexes):
        i = 0
        test_case_vector_length = \
            self.cycle_logs.get_test_case_vector_length(self.test_cases_vector[current_indexes[0]], self.conf.win_size)
        temp_obs = np.zeros((2, test_case_vector_length))

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
        self.width = 1
        self.right = 1
        self.left = 0
        self.end = 2
        self.index = 0
        self.current_obs = self.get_pair_data(self.current_indexes)
        self.test_cases_vector_temp = []
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
        elif selected_test_case['last_exec_time'] <= no_selected_test_case['last_exec_time']:
            reward = 0.5
        elif selected_test_case['last_exec_time'] > no_selected_test_case['last_exec_time']:
            reward = 0
        return reward

    def swapPositions(self, l, pos1, pos2):
        l[pos1], l[pos2] = l[pos2], l[pos1]
        return l

    def step(self, test_case_index):
        reward = self._calculate_reward(test_case_index)
        done = False
        if test_case_index == 1:
            self.test_cases_vector_temp.append(self.test_cases_vector[self.right])
            self.right = self.right+1
            if self.right >= self.end:
                while self.left < self.index+self.width:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
                    self.left = self.left+1
        elif test_case_index == 0:
            self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
            self.left = self.left + 1
            if self.left >= self.index+self.width:
                while self.right < self.end:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.right])
                    self.right = self.right+1

        if self.right < self.end and self.left < self.index+self.width:
            None
        elif self.end < len(self.test_cases_vector)-1:
            self.index = min(self.index+self.width*2, len(self.test_cases_vector)-1)
            self.left = self.index
            self.right = min(self.index + self.width, len(self.test_cases_vector)-1)
            self.end = min(self.right+self.width, len(self.test_cases_vector))
            if self.right < self.left+self.width:
                while self.left < self.end:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
                    self.left = self.left+1
                self.width = self.width * 2
                self.test_cases_vector = self.test_cases_vector_temp.copy()
                self.test_cases_vector_temp = []
                self.index = 0
                self.left = self.index
                self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
                self.end = min(self.right + self.width, len(self.test_cases_vector))
        elif self.width < len(self.test_cases_vector)/2:
            self.width = self.width*2
            self.test_cases_vector = self.test_cases_vector_temp.copy()
            self.test_cases_vector_temp = []
            self.index = 0
            self.left = self.index
            self.right = min(self.left + self.width, len(self.test_cases_vector)-1)
            self.end = min(self.right+self.width, len(self.test_cases_vector))
        else:
            done = True
            ## a2c reset the env when the epsiode is done, so we need to copy the result of test cases
            self.test_cases_vector = self.test_cases_vector_temp.copy()
            self.sorted_test_cases_vector = self.test_cases_vector.copy()
            return self.current_obs, reward, done, {}

        if not done:
            self.current_indexes[0] = self.left
            self.current_indexes[1] = self.right
            self.current_obs = self._next_observation(test_case_index)
        return self.current_obs, reward, done, {}
