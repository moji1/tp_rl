import pandas as pd

from testCase_prioritization.ci_cycle import CICycleLog


class TestCaseExecutionDataLoader:

    def __init__(self, data_path, data_format):
        self.data_path = data_path
        self.data_format = data_format
        self.test_data = None

    def load_data(self):
        df = pd.read_csv(self.data_path, error_bad_lines=False, sep=";")
        last_results = []
        max_size = 0
        ### process last result
        for i in range(df.shape[0]):
            last_result_str: str = df["LastResults"][i]
            temp_list = (last_result_str.strip("[").strip("]").split(","))
            if temp_list[0] != '':
                last_results.append(list(map(int, temp_list)))
            else:
                last_results.append([])
        df["LastResults"] = last_results
        self.test_data = df
        return self.test_data

    def pre_process(self):
        ## find max and min cycle id
        min_cycle = min(self.test_data["Cycle"])
        max_cycle = max(self.test_data["Cycle"])
        ci_cycle_logs = []
        ### process all cycles and save them in a list of CiCycleLohs
        for i in range(min_cycle, max_cycle + 1):
            ci_cycle_log = CICycleLog(i)
            cycle_rew_data = self.test_data.loc[self.test_data['Cycle'] == i]
            for index, test_case in cycle_rew_data.iterrows():
                ci_cycle_log.add_test_case(test_case["Id"], test_case["Name"], test_case["Duration"],
                                           test_case["Duration"],
                                           test_case["Verdict"], test_case["LastResults"], None)
            ci_cycle_logs.append(ci_cycle_log)
        return ci_cycle_logs
