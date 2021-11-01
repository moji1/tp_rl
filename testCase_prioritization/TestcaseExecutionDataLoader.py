import pandas as pd

from ci_cycle import CICycleLog


class TestCaseExecutionDataLoader:
    complexity_metric_list = ["AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","AvgEssential","AvgLine",
                            "AvgLineBlank","AvgLineCode","AvgLineComment","CountDeclClass","CountDeclClassMethod",
                            "CountDeclClassVariable","CountDeclExecutableUnit","CountDeclFunction",
                            "CountDeclInstanceMethod",
                            "CountDeclInstanceVariable","CountDeclMethod","CountDeclMethodDefault",
                            "CountDeclMethodPrivate",
                            "CountDeclMethodProtected","CountDeclMethodPublic","CountLine","CountLineBlank","CountLineCode",
                            "CountLineCodeDecl","CountLineCodeExe","CountLineComment","CountSemicolon","CountStmt",
                            "CountStmtDecl",
                            "CountStmtExe","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential",
                            "MaxNesting","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified",
                            "SumCyclomaticStrict","SumEssential"]
    def __init__(self, data_path, data_format):
        self.data_path = data_path
        self.data_format = data_format
        self.test_data = None

    def load_data(self):

        last_results = []
        cycle_ids = []
        max_size = 0
        ### process last result
        if self.data_format == "simple":
            df = pd.read_csv(self.data_path, error_bad_lines=False, sep=",")
            for i in range(df.shape[0]):
                last_result_str: str = df["LastResults"][i]
                temp_list = (last_result_str.strip("[").strip("]").split(","))
                if temp_list[0] != '':
                    last_results.append(list(map(int, temp_list)))
                else:
                 last_results.append([])
            df["LastResults"] = last_results
            self.test_data = df
        elif self.data_format == "enriched":
            df = pd.read_csv(self.data_path, error_bad_lines=False, sep=",")
            #df = df.rename(columns={'test_class_name': 'Id', 'time': 'last_exec_time',
            #                        'current_failures': 'Verdict'}, inplace=True)
            previous_cycle_commit = df["cycle_id"][0]
            cycle_id = 1
            for i in range(df.shape[0]):
                last_result = []
                last_result.append(df["failures_0"][i])
                last_result.append(df["failures_1"][i])
                last_result.append(df["failures_2"][i])
                last_result.append(df["failures_3"][i])
                last_results.append(last_result)
                if df["cycle_id"][i] != previous_cycle_commit:
                    assert len(df.loc[df['cycle_id'] == previous_cycle_commit]) == cycle_ids.count(cycle_id)
                    previous_cycle_commit = df["cycle_id"][i]
                    cycle_id = cycle_id + 1

                cycle_ids.append(cycle_id)


            df["LastResults"] = last_results
            df["Cycle"] = cycle_ids
            self.test_data = df
        return self.test_data

    def pre_process(self):
        ## find max and min cycle id

        min_cycle = min(self.test_data["Cycle"])
        max_cycle = max(self.test_data["Cycle"])
        ci_cycle_logs = []
        ### process all cycles and save them in a list of CiCycleLohs
        if self.data_format=='simple':
            for i in range(min_cycle, max_cycle + 1):
                ci_cycle_log = CICycleLog(i)
                cycle_rew_data = self.test_data.loc[self.test_data['Cycle'] == i]
                for index, test_case in cycle_rew_data.iterrows():
                    ci_cycle_log.add_test_case(test_id=test_case["Id"], test_suite=test_case["Name"],
                                               avg_exec_time=test_case["Duration"],
                                               last_exec_time=test_case["Duration"],
                                               verdict=test_case["Verdict"],
                                               failure_history=test_case["LastResults"],
                                               cycle_id=test_case["Cycle"],
                                               duration_group=test_case["DurationGroup"],
                                               time_group=test_case["TimeGroup"],
                                               exec_time_history=None)
                ci_cycle_logs.append(ci_cycle_log)
        elif self.data_format == 'enriched':
            for i in range(min_cycle, max_cycle + 1):
                ci_cycle_log = CICycleLog(i)
                cycle_rew_data = self.test_data.loc[self.test_data['Cycle'] == i]
                for index, test_case in cycle_rew_data.iterrows():
                    #add_test_case_enriched(self, test_id, test_suite, last_exec_time, verdict, avg_exec_time,
                    #                       failure_history=[], rest_hist=[], complexity=[]):
                    rest_hist=[]
                    rest_hist.append(test_case["failures_%"])
                    rest_hist.append(test_case["time_since"])
                    rest_hist.append(test_case["tests"])
                    complexity_metrics=[]
                    for metric in TestCaseExecutionDataLoader.complexity_metric_list:
                        complexity_metrics.append(test_case[metric])
                    ci_cycle_log.add_test_case_enriched(test_id=test_case["test_class_name"],
                                                        test_suite=test_case["test_class_name"],
                                                        last_exec_time=test_case["time"],
                                                        verdict=test_case["current_failures"],
                                                        avg_exec_time=test_case["time_0"],
                                                        failure_history=test_case["LastResults"],
                                                        rest_hist=rest_hist,
                                                        complexity_metrics=complexity_metrics,
                                                        cycle_id=test_case["cycle_id"])
                ci_cycle_logs.append(ci_cycle_log)

        return ci_cycle_logs
