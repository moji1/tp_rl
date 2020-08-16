import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime


from TPAgentUtil import TPAgentUtil
from PairWiseEnv import CIPairWiseEnv
from TPPairWiseDQNAgent import TPPairWiseDQNAgent
from ci_cycle import CICycleLog
from Config import Config
from TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from CustomCallback import  CustomCallback
from stable_baselines.bench import Monitor
from pathlib import Path

from testCase_prioritization.CIListWiseEnv import CIListWiseEnv
from testCase_prioritization.PointWiseEnv import CIPointWiseEnv


def train(trial, model):
    pass

def test(model):
    pass

def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


## find the cycle with maximum number of test cases
def get_max_test_cases_count(cycle_logs:[]):
    max_test_cases_count = 0
    for cycle_log in cycle_logs:
        if cycle_log.get_test_cases_count()>max_test_cases_count:
            max_test_cases_count = cycle_log.get_test_cases_count()
    return max_test_cases_count


def experiment(mode, algo, test_case_data, start_cycle, end_cycle, episodes, model_path, dataset_name, conf,verbos=False):
    log_dir = os.path.dirname(conf.log_file)
#    -- fix end cycle issue
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if start_cycle <= 0:
        start_cycle = 0
    if end_cycle >= len(test_case_data)-1:
        end_cycle = len(test_case_data)
    ## check for max cycle and end_cycle and set end_cycle to max if it is larger than max
    log_file = open(conf.log_file, "a")
    log_file_test_cases = open(log_dir+"/sorted_test_case.csv", "a")
    log_file.write("timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,winsize,test_cases,failed_test_cases, apfd, nrpa, random_apfd, optimal_apfd" + os.linesep)
    first_round: bool = True
    model_save_path = None
    for i in range(start_cycle, end_cycle - 1):
        if test_case_data[i].get_test_cases_count() < 6: # or test_case_data[i].get_failed_test_cases_count() < 1:
            continue
        if mode.upper() == 'PAIRWISE':
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * ((N * (N-1))/2))
            env = CIPairWiseEnv(test_case_data[i], conf)
        elif mode.upper() == 'POINTWISE':
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes) * N
            env = CIPointWiseEnv(test_case_data[i], conf)
        elif mode.upper() == 'LISTWISE':
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes) * get_max_test_cases_count(test_case_data)
            env = CIListWiseEnv(test_case_data[i], conf)
        print("Training agent with replaying of cycle " + str(i) + " with steps " + str(steps))

        if model_save_path:
            previous_model_path = model_save_path
        model_save_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
            start_cycle) + "_" + str(i)
        env = Monitor(env, model_save_path +"_monitor.csv")
        callback_class = CustomCallback(svae_path=model_save_path,
                                        check_freq=int(steps/episodes), log_dir=log_dir, verbose=verbos)

        if first_round:
            tp_agent = TPAgentUtil.create_model(algo, env)
            training_start_time = datetime.now()
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            training_end_time = datetime.now()
            first_round = False
        else:
            tp_agent = TPAgentUtil.load_model(algo=algo, env=env, path=previous_model_path+".zip")
            training_start_time = datetime.now()
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            training_end_time = datetime.now()
        print("Training agent with replaying of cycle " + str(i) + " is finished")

        j = i+1
        while (test_case_data[j].get_test_cases_count() < 6)  and j < end_cycle:\
            #or test_case_data[j].get_failed_test_cases_count() == 0) \
            j = j+1
        if j >= end_cycle-1:
            break
        if mode.upper() == 'PAIRWISE':
            env_test = CIPairWiseEnv(test_case_data[j], conf)
        elif mode.upper() == 'POINTWISE':
            env_test = CIPointWiseEnv(test_case_data[j], conf)
        elif mode.upper() == 'LISTWISE':
            env_test = CIListWiseEnv(test_case_data[j], conf)

        test_time_start = datetime.now()
        test_case_vector = TPAgentUtil.test_agent(env=env_test, algo=algo, model_path=model_save_path+".zip", mode=mode)
        test_time_end = datetime.now()
        test_case_id_vector = []

        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case['test_id']))
            cycle_id_text = test_case['cycle_id']
        if test_case_data[j].get_failed_test_cases_count() != 0:
            apfd = test_case_data[j].calc_APFD_ordered_vector(test_case_vector)
            apfd_optimal = test_case_data[j].calc_optimal_APFD()
            apfd_random = test_case_data[j].calc_random_APFD()
        else:
            apfd =0
            apfd_optimal =0
            apfd_random =0
        nrpa = test_case_data[j].calc_NRPA_vector(test_case_vector)
        test_time = millis_interval(test_time_start,test_time_end)
        training_time = millis_interval(training_start_time,training_end_time)
        print("Testing agent  on cycle " + str(j) +
              " resulted in APFD: " + str(apfd) +
              " , NRPA: " + str(nrpa) +
              " , optimal APFD: " + str(apfd_optimal) +
              " , random APFD: " + str(apfd_random) +
              " , # failed test cases: " + str(test_case_data[j].get_failed_test_cases_count()) +
              " , # test cases: " + str(test_case_data[j].get_test_cases_count()), flush=True)
        log_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(training_time) +
                       "," + str(test_time) + "," + str(conf.win_size) + "," +
                       str(test_case_data[j].get_test_cases_count()) + "," +
                       str(test_case_data[j].get_failed_test_cases_count()) + "," + str(apfd) + "," +
                       str(nrpa) + "," + str(apfd_random) + "," + str(apfd_optimal) + os.linesep)
        log_file_test_cases.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(training_time) +
                       "," + str(test_time) + "," + str(conf.win_size) + "," +
                                  ('|'.join(test_case_id_vector)) + os.linesep)
        log_file.flush()
        log_file_test_cases.flush()
    log_file.close()
    log_file_test_cases.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    # parser.add_argument('--traningData',help='tranind data folder',required=False)
    parser.add_argument('-m', '--mode', help='[pairwise,pointwise,listwise] ', required=True)
    parser.add_argument('-a', '--algo', help='[a2c,dqn,..]', required=True)
    parser.add_argument('-d', '--dataset_type', help='simple, enriched', required=False, default="simple")
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=True)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=True)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)


    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    supported_formalization = ['PAIRWISE', 'POINTWISE', 'LISTWISE']
    supported_algo = ['DQN', 'PPO2', "A2C", "ACKTR", "DDPG", "ACER", "GAIL", "HER", "PPO1", "SAC", "TD3", "TRPO"]
    args = parser.parse_args()
    assert supported_formalization.count(args.mode.upper()) == 1, "The formalization mode is not set correctly"
    assert supported_algo.count(args.algo.upper()) == 1, "The formalization mode is not set correctly"

    conf = Config()
    conf.train_data = args.train_data
    conf.dataset_name = Path(args.train_data).stem
    if not args.win_size:
        conf.win_size = 10
    else:
        conf.win_size = int(args.win_size)
    if not args.first_cycle:
        conf.first_cycle = 0
    else:
        conf.first_cycle = int(args.first_cycle)
    if not args.cycle_count:
        conf.cycle_count = 9999999

    if not args.output_path:
        conf.output_path = '../experiments/' + args.mode + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + args.mode + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"
    else:
        conf.output_path = args. output_path + "/" + args.mode + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + args.mode + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"

test_data_loader = TestCaseExecutionDataLoader(conf.train_data, args.dataset_type)
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()
### open data


# training using n cycle staring from start cycle
experiment(mode=args.mode, algo=args.algo.upper(), test_case_data=ci_cycle_logs, episodes=int(args.episodes),
           start_cycle=conf.first_cycle, verbos=False,
           end_cycle=conf.first_cycle + conf.cycle_count - 1, model_path=conf.output_path, dataset_name="", conf=conf)
# .. lets test this tommorow by passing args
