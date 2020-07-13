import numpy as np
from sklearn import preprocessing
import random


class CICycleLog:
    test_cases = {}
    cycle_id = 0

    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.test_cases = []

    def add_test_case(self, test_id, test_suite, avg_exec_time: int, last_exec_time: int, verdict: int,
                      failure_history: list, exec_time_history: list):
        test_case:dict = {}
        test_case['test_id'] = test_id
        test_case['test_suite'] = test_suite
        test_case['avg_exec_time'] = avg_exec_time
        test_case['verdict'] = verdict
        test_case['last_exec_time'] = last_exec_time
        if failure_history:
            test_case['failure_history'] = failure_history
            test_case['age'] = len(failure_history)
        else:
            test_case['failure_history'] = []
            test_case['age'] = 0
        if exec_time_history:
            test_case['exec_time_history'] = exec_time_history
        self.test_cases.append(test_case)

    def rem_test_case(self, test_id):
        if self.test_cases[test_id]:
            del self.test_cases[test_id]

    def export_test_cases(self, option: str, pad_digit=9, param1=0, param2=0, param3=0):
        if option == "list_avg_exec_with_failed_history":
            # assume param1 refers to the number of test cases,
            # params 2 refers to the history windows size, and param3 refers to pa
            test_cases_array = np.zeros((param1, param2 + 2))
            i = 0
            for test_case in self.test_cases:
                for j in range(0, len(test_case['failure_history'])):
                    if j >= param2:
                        break
                    test_cases_array[i][j] = test_case['failure_history'][j]
                for j in range(len(test_case['failure_history']), param2):
                    test_cases_array[i][j] = pad_digit
                test_cases_array[i][param2] = test_case['avg_exec_time']
                test_cases_array[i][param2 + 1] = len(test_case['failure_history'])
                i = i + 1
            for i in range(len(self.test_cases), param1):
                for j in range(0, param2 + 2):
                    test_cases_array[i][j] = pad_digit
            test_cases_array = preprocessing.normalize(test_cases_array, axis=0, norm='max')
            #test_cases_array[:, 1] = preprocessing.normalize(test_cases_array[:, 1])
            return test_cases_array
        else:
            return None

    def export_test_case(self, test_case: dict, option: str, pad_digit=9, win_size=0):
        if option == "list_avg_exec_with_failed_history":
            # assume param1 refers to the number of test cases,
            # params 2 refers to the history windows size, and param3 refers to pa
            test_case_vector = np.zeros((win_size + 2))
            for j in range(0, len(test_case['failure_history'])):
                if j >= win_size:
                    break
                test_case_vector[j] = test_case['failure_history'][j]
            for j in range(len(test_case), win_size):
                test_case_vector[j] = pad_digit
            test_case_vector[win_size] = test_case['avg_exec_time']
            test_case_vector[win_size + 1] = len(test_case)
            #test_cases_array = preprocessing.normalize(test_cases_vector, axis=0, norm='max')
            #test_cases_array[:, 1] = preprocessing.normalize(test_cases_array[:, 1])
            return test_case_vector
        else:
            return None

    def calc_APFD_vector_porb(self, test_case_vector_prob: list, threshold: float):
        sum_ranks: float = 0
        apfd: float = 0
        i = 1
        test_case_vector_prob = sorted(test_case_vector_prob, key=lambda x: x['prob'])
        for test_case_prob in test_case_vector_prob:
            sum_ranks = sum_ranks + self.test_cases[test_case_prob['index']]['verdict'] * i
            i = i+1
        N: float = self.get_test_cases_count()
        M: float = self.get_failed_test_cases_count()
        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))
        return apfd

    def calc_APFD_ordered_vector(self, test_case_vector: list):
        sum_ranks: float = 0
        apfd: float = 0
        i = 1
        for test_case in test_case_vector:
            sum_ranks = sum_ranks + test_case['verdict'] * i
            i = i+1
        N:float = self.get_test_cases_count()
        M:float = self.get_failed_test_cases_count()
        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))
        return apfd

    def calc_APFD(self, ordered_test_cases_id):
        sum_ranks: float = 0
        apfd: float = 0
        ordered_test_cases_temp=[]
        for test_case_id in ordered_test_cases_id:
            if test_case_id < self.get_test_cases_count():
                ordered_test_cases_temp.append(test_case_id)
        ordered_test_cases = ordered_test_cases_temp
        for i in range(0, len(ordered_test_cases)):
            if ordered_test_cases[i]< self.get_test_cases_count():
                sum_ranks = sum_ranks + self.test_cases[ordered_test_cases[i]]['verdict'] * (i + 1)
        N:float = self.get_test_cases_count()
        M:float = self.get_failed_test_cases_count()
        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))
        return apfd

    def calc_random_APFD(self):
        random_order = []
        while len(random_order) < self.get_test_cases_count():
            rand_num = random.randint(0, self.get_test_cases_count())
            if random_order.count(rand_num) <= 0:
                random_order.append(rand_num)
        random_apfd = self.calc_APFD(random_order)
        return random_apfd

    def calc_optimal_APFD(self):
        optimal_order = sorted(self.test_cases, key=lambda x: x['verdict'], reverse=True)
        sum_ranks = 0
        i = 1
        apfd: float = 0
        for test_case in optimal_order:
            sum_ranks = sum_ranks + test_case['verdict'] * i
            i = i+1
        N: float = self.get_test_cases_count()
        M: float = self.get_failed_test_cases_count()
        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))
        return apfd

    def get_failed_test_cases_count(self):
        cnt = 0
        for test_case in self.test_cases:
            if test_case['verdict'] == 1:
                cnt = cnt + 1
        return cnt

    def get_test_cases_count(self) -> object:
        return len(self.test_cases)

    def get_passed_test_cases_count(self):
        cnt = 0
        for test_case in self.test_cases:
            if test_case['verdict'] == 0:
                cnt = cnt + 1
        return cnt

    def get_max_last_exec_time(self):
        return max(self.test_cases, key=lambda x: x['last_exec_time'])['last_exec_time']

    def get_min_last_exec_time(self):
        return min(self.test_cases, key=lambda x: x['last_exec_time'])['last_exec_time']

    def get_test_case_last_exec_time(self, test_case_index: int):
        return self.test_cases[test_case_index]['last_exec_time']

    def get_test_case_last_exec_time_normalized(self, test_case_index: int):
        last_exec_time: int = self.get_test_case_last_exec_time(test_case_index)
        min_last_exec_time = self.get_min_last_exec_time()
        max_last_exec_time = self.get_max_last_exec_time()
        if max_last_exec_time-min_last_exec_time>0:
            last_exec_time_norm: float = (last_exec_time - min_last_exec_time)/(max_last_exec_time-min_last_exec_time)
        else:
            last_exec_time_norm = 0
        return last_exec_time_norm

    def get_test_case_verdict(self, test_case_index: int):
        return self.test_cases[test_case_index]['verdict']
