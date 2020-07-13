import argparse
import pandas as pd
import numpy as np
import os



from testCase_prioritization.PointWiseEnv import CIPointWiseEnv
from testCase_prioritization.TPPointWiseAgent import TPPointWisePPO2Agent
from testCase_prioritization.ci_cycle import CICycleLog
from testCase_prioritization.Config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    # parser.add_argument('--traningData',help='tranind data folder',required=False)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=False)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-n', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-m', '--max_list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)

    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    conf = Config()
    args = parser.parse_args()
    if not args.win_size:
        conf.win_size = 5
    if not args.first_cycle:
        conf.first_cycle = 1
    if not args.cycle_count:
        conf.cycle_count = 350
    if not args.train_data:
        conf.train_data = '../data/tc_data_paintcontrol.csv'
    if not args.output_path:
        conf.output_path = '../data/PPO2Agent'

### open data
df = pd.read_csv(conf.train_data, error_bad_lines=False, sep=";")
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

## find max and min cycle id
min_cycle = min(df["Cycle"])
max_cycle = max(df["Cycle"])
ci_cycle_logs = []

### process all cycles and save them in a list of CiCycleLohs
for i in range(min_cycle, max_cycle + 1):
    ci_cycle_log = CICycleLog(i)
    cycle_rew_data = df.loc[df['Cycle'] == i]
    for index, test_case in cycle_rew_data.iterrows():
        ci_cycle_log.add_test_case(test_case["Id"], test_case["Name"], test_case["Duration"], test_case["Duration"],
                                   test_case["Verdict"], test_case["LastResults"], None)
    ci_cycle_logs.append(ci_cycle_log)

# training using n cycle staring from start cycle

first_round = True
tp_agent = TPPointWisePPO2Agent()
f = open("../models/evaluation_pointwise_10000_paint_controller_winsize_5_logs.txt", "a")
f.write("cycle_id,test_cases,failed_test_cases,apfd,random_apfd,optimal_apfd" + os.linesep)
for i in range(conf.first_cycle, conf.first_cycle + conf.cycle_count - 1):
    if ci_cycle_logs[i].get_test_cases_count() <= 1 or ci_cycle_logs[i].get_failed_test_cases_count() <= 1:
        continue
    print("Training agent with replaying of cycle " + str(i))
    env = CIPointWiseEnv(ci_cycle_logs[i], conf)
    if first_round:
        model = tp_agent.train_agent(env, 100000, conf.output_path+"_pointwise_10000_paint_controller_winsize_5_" +
                                     str(conf.first_cycle) + "_" + str(i))
        first_round = False
    else:
        model = tp_agent.train_agent(env=env, steps=100000,
                                     path_to_save_agent=conf.output_path+"_pointwise_10000_paint_controller_winsize_5_"
                                                        + str(conf.first_cycle) + "_" + str(i),
                                     base_model=model)
    print("Training agent with replaying of cycle " + str(i) + " is finished")
    # check_env(env)
    ### test the model on the next cycle
    if ci_cycle_logs[i + 1].get_test_cases_count() < 2:
        continue
    env_test = CIPointWiseEnv(ci_cycle_logs[i + 1], conf)
    try:
        # agent_actions = tp_agent.test_agent(env=env_test, model=None,
        #                                 model_path="DQN_prioritization_" + str(args.first_cycle) + "_" + str(i))

        test_case_vector_prob = tp_agent.test_agent(env=env_test, model=model,
                                            model_path="DQN_prioritization_" + str(conf.first_cycle) + "_" + str(i))
    except TimeoutError:
        print("Testing agent on cycle " + str(i + 1) + " timed out")
        continue
    apfd = ci_cycle_logs[i + 1].calc_APFD_vector_porb(test_case_vector_prob, .80)
    apfd_optimal = ci_cycle_logs[i + 1].calc_optimal_APFD()
    apfd_random = ci_cycle_logs[i + 1].calc_random_APFD()
    print("Testing agent  on cycle " + str(i + 1) + " resulted in APFD: " + str(apfd) +
          " , optimal APFD: " + str(apfd_optimal) +
          " , random APFD: " + str(apfd_random) +
          " , # failed test cases: " + str(ci_cycle_logs[i + 1].get_failed_test_cases_count()) +
          " , # test cases: " + str(ci_cycle_logs[i + 1].get_test_cases_count()))
    f.write(str(i + 1) + "," + str(ci_cycle_logs[i + 1].get_test_cases_count()) + "," +
            str(ci_cycle_logs[i + 1].get_failed_test_cases_count()) + "," + str(apfd) + "," +
            str(apfd_random) + "," + str(apfd_optimal) + os.linesep)
    f.flush()
f.close()
