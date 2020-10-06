from typing import Any, Union

import numpy as np
import gym
from gym import spaces
from sklearn import preprocessing
from ci_cycle import CICycleLog
from Config import Config


class CIPointWiseEnv(gym.Env):
    def __init__(self, cycle_logs: CICycleLog, conf: Config):
        super(CIPointWiseEnv, self).__init__()
        self.conf = conf
        self.reward_range = (-1, 1)
        self.cycle_logs = cycle_logs
        self.test_cases_vector_prob = []
        self.current_index = 0
        self.optimal_order= cycle_logs.get_optimal_order()
        self.testcase_vector_size = self.cycle_logs.get_test_case_vector_length(cycle_logs.test_cases[0],self.conf.win_size)
        self.current_obs = np.zeros((1, self.testcase_vector_size))
        self.initial_observation = self.get_point_data(self.current_index)
        self.current_obs = self.initial_observation.copy()

        # self.number_of_actions = len(self.cycle_logs.test_cases)
        #self.action_space = spaces.discrete()
        self.action_space = spaces.Box(low=0, high=1, shape=(1, ))
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(1, self.testcase_vector_size))  # ID, execution time and LastResults

    def get_point_data(self, test_case_index):
        temp_obs = self.cycle_logs.export_test_case(self.cycle_logs.test_cases[test_case_index],
                                                    "list_avg_exec_with_failed_history",
                                                    self.conf.padding_digit,
                                                    self.conf.win_size)
        return temp_obs

    def render(self, mode='human'):
        pass

    def reset(self):
        self.test_cases_vector_prob = []
        self.current_index = 0
        self.current_obs = self.get_point_data(self.current_index)
        return self.current_obs

    def _next_observation(self, index):
        self.current_obs = self.get_point_data(self.current_index)
        return self.current_obs

    def _initial_obs(self):
        return self.initial_observation

    ## the reward function must be called before updating the observation
    def _calculate_reward(self, test_case_prob):
        test_case_prob = test_case_prob[0]
        optimal_rank= self.optimal_order.index(self.cycle_logs.test_cases[self.current_index])
        normalized_optimal_rank=optimal_rank/self.cycle_logs.get_test_cases_count()
        reward = 1 - (test_case_prob-normalized_optimal_rank)**2
        return reward
    def _calculate_reward_old1(self, test_case_prob):
        test_case_prob = test_case_prob[0]
        if self.cycle_logs.test_cases[self.current_index]['verdict']:
            if test_case_prob < .80:
                return -1 * (0.80 - test_case_prob)
            else:
                return test_case_prob
        else:
            if test_case_prob > .20:
                return -1 * (test_case_prob - 0.2)
            else:
                return 1 - test_case_prob

    def step(self, test_case_prob):
        done = False
        reward = self._calculate_reward(test_case_prob)
        test_case_prob = {'index': self.current_index, 'prob': test_case_prob}
        self.test_cases_vector_prob.append(test_case_prob)
        self.current_index = self.current_index + 1
        if self.current_index < self.cycle_logs.get_test_cases_count():
            self.current_obs = self._next_observation(self.current_index)
        else:
            done = True
            self.current_obs = np.zeros(self.testcase_vector_size)
        return self.current_obs, reward, done, {}
