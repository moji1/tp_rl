import argparse
import pandas as pd
import numpy as np
import os

from testCase_prioritization.PairWiseEnv import CIPairWiseEnv
from testCase_prioritization.TPPairWiseA2CAgent import TPPairWiseA2CAgent
from testCase_prioritization.TPPairWiseDQNAgent import TPPairWiseDQNAgent
from testCase_prioritization.ci_cycle import CICycleLog
from testCase_prioritization.Config import Config
from util.TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from testCase_prioritization.CustomCallback import  CustomCallback
from stable_baselines.bench import Monitor




def experiment(mode, algorithm, test_case_data, start_cycle, end_cycle, episodes, model_path, dataset_name, conf):
    log_dir = os.path.dirname(conf.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ## check for max cycle and end_cycle and set end_cycle to max if it is larger than max

    log_file = open(conf.log_file, "a")
    log_file.write("cycle_id,test_cases,failed_test_cases, apfd, random_apfd, optimal_apfd" + os.linesep)
    first_round: bool = True
    if mode.lower() == 'pairwise':
        if algorithm.lower() == "dqn":
            tp_agent = TPPairWiseDQNAgent()
        elif algorithm.lower() == "a2c":
            tp_agent = TPPairWiseA2CAgent()
        elif  algorithm.lower() == "a2c":

    for i in range(start_cycle, end_cycle - 1):
        if test_case_data[i].get_test_cases_count() <= 1 or test_case_data[i].get_failed_test_cases_count() <= 1:
            continue
        if mode == 'pairwise':
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * ((N * (N-1))/2))
            env = CIPairWiseEnv(test_case_data[i], conf)
        print("Training agent with replaying of cycle " + str(i) + " with steps " + str(steps))
        env = Monitor(env, log_dir)
        model_save_path = model_path + "/" + mode + "_" + algorithm + dataset_name + "_" + str(
            start_cycle) + "_" + str(i)
        callback_class = CustomCallback(svae_path=model_save_path,
                                        check_freq=int((N * (N-1))/2), log_dir=log_dir, verbose=1)

        if first_round:
            model = tp_agent.train_agent(env, steps,
                                         path_to_save_agent=model_save_path, callback_class=callback_class)

            first_round = False
        else:
            model = tp_agent.train_agent(env=env, steps=steps,
                                         path_to_save_agent=model_save_path, base_model=model,
                                         callback_class=callback_class)
        print("Training agent with replaying of cycle " + str(i) + " is finished")

        if test_case_data[i + 1].get_test_cases_count() < 2:
            continue
        if mode == 'pairwise':
            env_test = CIPairWiseEnv(test_case_data[i + 1], conf)

        test_case_vector = tp_agent.test_agent(env=env_test, model=None, model_path=model_save_path+".zip")
        apfd = test_case_data[i + 1].calc_APFD_ordered_vector(test_case_vector)
        apfd_optimal = test_case_data[i + 1].calc_optimal_APFD()
        apfd_random = test_case_data[i + 1].calc_random_APFD()
        print("Testing agent  on cycle " + str(i + 1) + " resulted in APFD: " + str(apfd) +
              " , optimal APFD: " + str(apfd_optimal) +
              " , random APFD: " + str(apfd_random) +
              " , # failed test cases: " + str(test_case_data[i + 1].get_failed_test_cases_count()) +
              " , # test cases: " + str(test_case_data[i + 1].get_test_cases_count()))
        log_file.write(str(i + 1) + "," + str(test_case_data[i + 1].get_test_cases_count()) + "," +
                       str(test_case_data[i + 1].get_failed_test_cases_count()) + "," + str(apfd) + "," +
                       str(apfd_random) + "," + str(apfd_optimal) + os.linesep)
        log_file.flush()
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    # parser.add_argument('--traningData',help='tranind data folder',required=False)
    parser.add_argument('-m', '--mode', help='Formalization mode ', required=True)
    parser.add_argument('-a', '--algo', help='Formalization mode ', required=True)
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=True)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=True)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)

    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    supported_formalization = ['pairwise', 'pointwise', 'listwise']
    supported_algo = ['DQN', 'PPO2', "A2C", "ACKTR", "DDPG", "ACER", "GAIL", "HER", "PPO1", "SAC", "TD3", "TRPO"]
    args = parser.parse_args()
    assert supported_formalization.count(args.mode.upper()) == 1, "The formalization mode is not set correctly"
    assert supported_algo.count(args.algo.upper()) == 1, "The formalization mode is not set correctly"

    conf = Config()
    conf.train_data = args.train_data

    if not args.win_size:
        conf.win_size = 5
    else:
        conf.win_size = int(args.win_size)
    if not args.first_cycle:
        conf.first_cycle = 1
    else:
        conf.first_cycle = int(args.first_cycle)
    if not args.cycle_count:
        conf.cycle_count = 9999999

    if not args.output_path:
        conf.output_path = '../experiments/' + args.mode + "/" + args.algo + "/"
        conf.log_file = '../experiments/' + args.mode + "/" + args.algo + "/" + args.mode + "_" + args.algo + "_" + args.episodes + "_log.txt"

test_data_loader = TestCaseExecutionDataLoader(conf.train_data, "simple1")
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()
### open data


# training using n cycle staring from start cycle
experiment(mode=args.mode, algorithm=args.algo, test_case_data=ci_cycle_logs, episodes=int(args.episodes),
           start_cycle=conf.first_cycle,
           end_cycle=conf.first_cycle + conf.cycle_count - 1, model_path=conf.output_path, dataset_name="", conf=conf)
# .. lets test this tommorow by passing args
