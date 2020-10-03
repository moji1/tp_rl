import argparse
import pandas as pd
import pingouin as pg

from pathlib import Path
import scipy.stats as stats
from statistics import mean
import collections






def save_to_db():
    None


def process_logs(datasets, supported_formalization, supported_algo):

    accuracy_dataset_result = {"iofrol-additional-features":{}, 'paintcontrol-additional-features': {},
                    "Commons_codec": {}, "Commons_io": {}, "Commons_imaging": {},
                   "Commons_compress": {}, "Commons_math": {}, "Commons_lang": {}}
    training_time_dataset_result = {"iofrol-additional-features":{}, 'paintcontrol-additional-features': {},
                    "Commons_codec": {}, "Commons_io": {}, "Commons_imaging": {},
                   "Commons_compress": {}, "Commons_math": {}, "Commons_lang": {}}

    accuracy_summary= {"A2C": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "ACER":  {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "ACKTR":  {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "DDPG":  {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "DQN": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "PPO1": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "PPO2": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "SAC": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "TD3": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "TRPO": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}}}

    training_time_summary = {"A2C": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "ACER":  {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "ACKTR":  {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "DDPG":  {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "DQN": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "PPO1": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "PPO2": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "SAC": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "TD3": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}},
                             "TRPO": {"PAIRWISE": {},"POINTWISE": {},"LISTWISE": {}}}
    for dataset_type in datasets.keys():
        for dataset in datasets[dataset_type]:
            for mode in supported_formalization:
                for algo in supported_algo[mode]:
                    log_file_path = Path(f"{log_path}//{mode}//{algo}//{dataset}_{winsize[dataset_type]}//" \
                                    f"{mode}_{algo}_{dataset}_{episode}_{winsize[dataset_type]}_log.txt")
                    if log_file_path.is_file() :
                        #print(log_file_path)
                        data = pd.read_csv(log_file_path, error_bad_lines=False, sep=",")
                        if len(data) == dataset_size[dataset]:
                            data.rename(columns=lambda x: x.strip(), inplace=True)
                            #print(log_file_path)
                            training_times = data["training_time"]
                            training_times_sum = training_times.sum()
                            if not (dataset in training_time_summary[algo][mode].keys()):
                                training_time_summary[algo][mode][dataset] = {}
                            training_time_summary[algo][mode][dataset]['sum'] = training_times_sum/(1000*60)
                            training_time_dataset_result[dataset][mode + "-" + algo] = training_times
                            if not (dataset in accuracy_summary[algo][mode].keys()):
                                accuracy_summary[algo][mode][dataset] = {}
                            if (dataset_type=="simple"):
                                apfds = data["apfd"]
                                accuracy_summary[algo][mode][dataset]["mean"] = apfds.mean(skipna = True)
                                accuracy_summary[algo][mode][dataset]["std"] = apfds.std(skipna=True)
                                accuracy_dataset_result[dataset][mode+"-"+algo] = apfds
                                #result = f"{mode}_{algo}_{dataset}: \small {(apfds.mean(skipna = True)):.3f}$\pm${apfds.std(skipna = True):.3f}" \
                                #     f" \circled(x)"
                            else:
                                nrpas = data["nrpa"]
                                accuracy_summary[algo][mode][dataset]["mean"] = nrpas.mean(skipna = True)
                                accuracy_summary[algo][mode][dataset]["std"] = nrpas.std(skipna = True)
                                accuracy_dataset_result[dataset][mode+"-"+algo] = nrpas
                                #result = f"{mode}_{algo}_{dataset}: \small {(nrpas.mean(skipna = True)):.3f}$\pm${nrpas.std(skipna = True):.3f}" \
                                #     f" \circled(x)"
                        else:
                            result = f"{log_file_path}: data size issue"
                    else:
                        result = f"{log_file_path}: logs is not availabe yet"
                        print(result)

    return accuracy_dataset_result, accuracy_summary,training_time_dataset_result,training_time_summary

def anova_analysis(df, reverse=True): # df is a melted dataframe
    aov = pg.welch_anova(dv='value', between='variable', data=df)
    #print(aov)
    all_configs = df['variable'].unique()
    ## check if aov pvalue is less than 0.05
    pairwise_data = pg.pairwise_gameshowell(data=df, dv='value', between='variable').round(3)
    gt_relation_positive = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] > 0.0)]["A"]
    lt_relation_negetive = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] < 0)]["B"]
    all_significant = pd.DataFrame(pd.concat([gt_relation_positive, lt_relation_negetive]), columns=["variable"])
    gt_relation_negetive = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] > 0.0)]["B"]
    lt_relation_postive = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] < 0)]["A"]
    all_not_significant = pd.DataFrame(pd.concat([gt_relation_negetive, lt_relation_postive]), columns=["variable"])
    positive_points = all_significant["variable"].value_counts()
    negetive_points = all_not_significant["variable"].value_counts()


    effect_sizes=[]
    ranks={}
    for config in all_configs:
        ranks[config] = 0
    for config in positive_points.iteritems():
        ranks[config[0]] += config[1]
    for config in negetive_points.iteritems():
        ranks[config[0]] -= config[1]
    ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=reverse)
    best_config = ranks[0][0]
    worst_config = ranks[len(ranks)-1][0]
    if isinstance(best_config, str) and isinstance(worst_config, str):
        effect_size = pg.compute_effsize(df[df["variable"] == best_config]["value"],
                       df[df["variable"] == worst_config]["value"], eftype='CLES')

    if reverse:
        prev_rank = 9999
    else:
        prev_rank = -9999
    real_rank = 0
    final_ranks = {}
    for rank in ranks:
        if (rank[1] > prev_rank and not reverse) or (rank[1] < prev_rank and reverse):
            prev_rank = rank[1]
            real_rank += 1
        final_ranks[rank[0]] = real_rank


    return final_ranks, aov["p-unc"][0] , best_config, worst_config, effect_size

def latex_accuracy(algo_mode,datasets,accuracy_summary):
    latex_text = f""
    for algo in algo_mode.keys():
        latex_text += f"\multirow{{{len(algo_mode[algo])}}}{{.40cm}}{{\small \\textbf{{{algo}}}}} "
        #\small \textbf {PO} & \small
        #.650$\pm$.260 \circled{6} &
        for mode in algo_mode[algo]:
            latex_text=latex_text+ f"& \small \\textbf{{{mode[0]+mode[1]}}} "
            for dataset_type in datasets.keys():
                for dataset in datasets[dataset_type]:
                    if not dataset in accuracy_summary[algo][mode].keys():
                        accuracy_summary[algo][mode][dataset] = {}
                        accuracy_summary[algo][mode][dataset]['mean'] = 0
                        accuracy_summary[algo][mode][dataset]['std'] = 0
                        accuracy_summary[algo][mode][dataset]['rank'] = 0
                    latex_text += f"& \small  {accuracy_summary[algo][mode][dataset]['mean']:.2f}$\pm$" \
                                  f"{accuracy_summary[algo][mode][dataset]['std']:.2f} " \
                                              f"\circled{{{accuracy_summary[algo][mode][dataset]['rank']}}} "
            latex_text=latex_text+ f"\\\\ \n"
        latex_text += f"\\hline \n"
    return  latex_text

def latex_training(algo_mode, datasets, training_time_summary):
    latex_text = f""
    for algo in algo_mode.keys():
        latex_text += f"\multirow{{{len(algo_mode[algo])}}}{{.40cm}}{{\small \\textbf{{{algo}}}}} "
        for mode in algo_mode[algo]:
            latex_text=latex_text+ f"& \small \\textbf{{{mode[0]+mode[1]}}} "
            for dataset_type in datasets.keys():
                for dataset in datasets[dataset_type]:
                    if not dataset in training_time_summary[algo][mode].keys():
                        training_time_summary[algo][mode][dataset] = {}
                        training_time_summary[algo][mode][dataset]['sum'] = 0
                        training_time_summary[algo][mode][dataset]['rank'] = 0
                    latex_text += f"& \small  {training_time_summary[algo][mode][dataset]['sum']:.1f} " \
                                  f"\\circled{{{training_time_summary[algo][mode][dataset]['rank']}}} "
            latex_text=latex_text+ f"\\\\ \n"
        latex_text += f"\\hline \n"
    return  latex_text

def calc_ttest(data1, data2):

    ttest = stats.ttest_ind(data1, data2, equal_var=False)
    effect_size = pg.compute_effsize(data1,
                                     data2, eftype='CLES')
    return  ttest, effect_size



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=True)
    parser.add_argument('-i', '--input', help='Location of log files', required=True)
    parser.add_argument('-e', '--episode', help='Number of episodes ', required=True)
    args = parser.parse_args()


    log_path = args.input
    supported_formalization = ['PAIRWISE', 'POINTWISE', 'LISTWISE']
    #supported_algo = ['DQN', 'PPO2', "A2C", "ACKTR", "DDPG", "ACER", "GAIL", "HER", "PPO1", "SAC", "TD3", "TRPO"]
    episode = args.episode
    supported_algo = {"PAIRWISE":  ['DQN', 'PPO2', "A2C", "ACKTR", "ACER", "PPO1", "TRPO"],
                      "LISTWISE":  ['DQN', 'PPO2', "A2C", "ACER", "PPO1", "TRPO"],
                      "POINTWISE": ['DDPG', 'PPO2', "A2C", "SAC", "TD3", "TRPO", "PPO1","ACKTR"]}

    algo_mode = {"A2C": ["PAIRWISE", "POINTWISE", "LISTWISE"],
                        "ACER": ["PAIRWISE", "LISTWISE"],
                        "ACKTR": ["PAIRWISE", "POINTWISE"],
                        "DDPG": ["POINTWISE"],
                        "DQN": ["PAIRWISE", "LISTWISE"],
                        "PPO1": ["PAIRWISE", "POINTWISE", "LISTWISE"],
                        "PPO2": ["PAIRWISE", "POINTWISE", "LISTWISE"],
                        "SAC": ["POINTWISE"],
                        "TD3": ["POINTWISE"],
                        "TRPO":  ["PAIRWISE", "POINTWISE", "LISTWISE"]}

    datasets = {"simple":   ['iofrol-additional-features', 'paintcontrol-additional-features'],
                "enriched": ["Commons_codec", "Commons_imaging", "Commons_io",
                            "Commons_compress","Commons_lang","Commons_math"]}
    ds_print = {'iofrol-additional-features':"IOFROL", 'paintcontrol-additional-features':"Paint.",
                          "Commons_codec":"CODEC", "Commons_imaging":"IMAG", "Commons_io":"IO",
                            "Commons_compress":"COMP","Commons_lang":"LANG","Commons_math":"MATH"}
    winsize={"simple":10,"enriched":4}
    winsize["simple"] = args.win_size

    dataset_size = {'iofrol-additional-features': 203, 'paintcontrol-additional-features': 251,
                    "Commons_codec": 177, "Commons_io": 175, "Commons_imaging": 145,
                   "Commons_compress": 436, "Commons_math": 54, "Commons_lang": 299}

    accuracy_dataset_result, accuracy_summary, training_time_dataset_result, training_time_summary = process_logs(datasets,supported_formalization,supported_algo)
    #data=pd.DataFrame(accuracy_dataset_result["Commons_codec"])
    #RQ1
    ## anova analysis for eachdata set
    ## Generate Accuracy  Table
    ## Generate Training time Table

    best_worst_config_latex=f"" ## CLE table latex
    for dataset_type in datasets.keys():
        for dataset in datasets[dataset_type]:
            ## accuracy analysis
            data = pd.DataFrame(accuracy_dataset_result[dataset])
            config_ranks, aov_pval, best_config, worst_config, max_CLE = anova_analysis(pd.melt(data),  reverse=True)
            best_worst_config_latex += f"\\small \\textbf{{{ds_print[dataset]}}} &\\small {best_config} & {worst_config}  & {max_CLE:.3f} \\\\ \n"
            for config in config_ranks:
                temp = config.split("-")
                mode = temp[0]
                algo = temp[1]
                accuracy_summary[algo][mode][dataset]["rank"] = config_ranks[config]
            ## training time analysis
            data = pd.DataFrame(training_time_dataset_result[dataset])
            config_ranks, aov_pval, best_config, worst_config, max_CLE = anova_analysis(pd.melt(data), reverse=False)
            for config in config_ranks:
                temp = config.split("-")
                mode = temp[0]
                algo = temp[1]
                training_time_summary[algo][mode][dataset]["rank"] = config_ranks[config]

    #generate_latex_table_accuracy(accuracy_summary)
    latex_accuracy_text = latex_accuracy(algo_mode,datasets,accuracy_summary)
    print("Table for accuracy of configurations:\n")
    print(latex_accuracy_text)
    print("\n\n")
    #generate_latex_table_trainingtime(training_time_summary)
    print("Table for training time: \n")
    latex_training_text = latex_training(algo_mode, datasets, training_time_summary)
    print(latex_training_text)
    print("\n\n")
    print("Table for best-worst cle:\n")
    print(best_worst_config_latex)

    ## RQ1.2
    print("RQ1.2:")
    shared_algos = ["PPO1", "PPO2", "TRPO", "A2C"]
    shared_algo_accuracy_result = {"PAIRWISE": [], "POINTWISE": [], "LISTWISE": []}
    shared_algo_train_time_result = {"PAIRWISE": [], "POINTWISE": [], "LISTWISE": []}
    #shared_algo_simple_result= prepare_resuts_of_shared_algo(accuracy_dataset_result)
    for dataset_type in datasets.keys():
        for share_algo in shared_algos:
            shared_algo_accuracy_result = {"PAIRWISE": [], "POINTWISE": [], "LISTWISE": []}
            shared_algo_train_time_result = {"PAIRWISE": [], "POINTWISE": [], "LISTWISE": []}
            for dataset in datasets[dataset_type]:
                for mode in supported_formalization:
                    if mode+"-"+share_algo in accuracy_dataset_result[dataset].keys():
                        shared_algo_accuracy_result[mode] += (accuracy_dataset_result[dataset][mode+"-"+share_algo].tolist())
                        shared_algo_train_time_result[mode] += (
                            training_time_dataset_result[dataset][mode + "-" + share_algo].tolist())

            if  (len(shared_algo_accuracy_result["PAIRWISE"]) == len(shared_algo_accuracy_result["LISTWISE"]) and
                len(shared_algo_accuracy_result["PAIRWISE"]) == len(shared_algo_accuracy_result["POINTWISE"]) ) :
                data = pd.DataFrame(shared_algo_accuracy_result)
                config_ranks, aov_pval, best_config, worst_config, max_CLE = anova_analysis(pd.melt(data),  reverse=True)
                results = f"Accuracy: {share_algo}:{dataset_type} pvalue={aov_pval}, max_cle={max_CLE} Rankes:{config_ranks}"
                print(results)
                data = pd.DataFrame(shared_algo_train_time_result)
                config_ranks, aov_pval, best_config, worst_config, max_CLE = anova_analysis(pd.melt(data),  reverse=False)
                results = f"Training Time: {share_algo}:{dataset_type} pvalue={aov_pval}, max_cle={max_CLE}, Rankes:{config_ranks}"
                print(results)


    ## RQ1.3
    print("RQ1.3")
    for dataset_type in datasets.keys():
        if dataset_type == "simple":
            supported_algo["LISTWISE"] = ['PPO2', "A2C", "ACER", "PPO1", "TRPO"]
        else:
            supported_algo["LISTWISE"] = ['DQN','PPO2', "A2C", "ACER", "PPO1", "TRPO"]
        for mode in supported_formalization:
            results_algo_accuracy = {}
            results_algo_training = {}
            for algo in supported_algo[mode]:
                for dataset in datasets[dataset_type]:
                    if not algo in results_algo_accuracy:
                        results_algo_accuracy[algo] = []
                        results_algo_training[algo] = []
                    if mode + "-" + algo in accuracy_dataset_result[dataset].keys():
                        results_algo_accuracy[algo] += (accuracy_dataset_result[dataset][mode + "-" + algo].tolist())
                        results_algo_training[algo] += (training_time_dataset_result[dataset][mode + "-" + algo].tolist())


            data = pd.DataFrame(results_algo_accuracy)
            config_ranks, aov_pval, best_config, worst_config, max_CLE = anova_analysis(pd.melt(data), reverse=True)
            results = f"Accuracy: {mode}-{dataset_type} pvalue={aov_pval}, max_cle={max_CLE} Rankes:{config_ranks}"
            print(results)
            data = pd.DataFrame(results_algo_training)
            config_ranks, aov_pval, best_config, worst_config, max_CLE = anova_analysis(pd.melt(data),
                                                                                        reverse=False)
            results = f"Training Time: {mode}-{dataset_type} pvalue={aov_pval}, max_cle={max_CLE}, Rankes:{config_ranks}"
            print(results)

    ### RQ2
    ## RQ2.1
    print("RQ1.2")
    best_config ="PAIRWISE-ACER"

    rl_bs1 = {"paintcontrol-additional-features": r"../experiments/RL-BS1/RLBS1-iofrol-additional-features.csv",
             "iofrol-additional-features": r"../experiments/RL-BS1/RLBS1-paintcontrol-additional-features.csv"}

    for dataset in rl_bs1.keys():
        log_file_path = rl_bs1[dataset]
        rl_bs1_data = pd.read_csv(log_file_path, error_bad_lines=False, sep=";")
        rl_bs1_result = rl_bs1_data['apfd'].dropna()
        pairwise_ACER_result = accuracy_dataset_result[dataset][best_config]
        ttest, effect_size = calc_ttest(pairwise_ACER_result, rl_bs1_result)
        result = f"RLBS1: {dataset} mean: {rl_bs1_result.mean(skipna=True)}, STD:{rl_bs1_result.std(skipna=True)}" \
                 f" pvalue: {ttest.pvalue:.6f}, CLE: {effect_size:.3f}"
        print(result)

    rl_bs2 = {"Commons_codec": r"../experiments/ICSE2020/BS2-Commons_codec.csv",
             "Commons_imaging": r"../experiments/ICSE2020/BS2-Commons_imaging.csv",
             "Commons_io": r"../experiments/ICSE2020/BS2-Commons_io.csv",
             "Commons_compress": r"../experiments/ICSE2020/BS2-Commons_compress.csv",
             "Commons_lang": r"../experiments/ICSE2020/BS2-Commons_lang.csv",
             "Commons_math": r"../experiments/ICSE2020/BS2-Commons_math.csv"}
    avg_diff = []
    avg_diff2=[]
    for dataset in rl_bs2:
        pairwise_ACER_result = accuracy_dataset_result[dataset][best_config]
        log_file_path = rl_bs2[dataset]
        rl_bs2_data = pd.read_csv(log_file_path, error_bad_lines=False, sep=",")
        rl_bs2_result = rl_bs2_data[rl_bs2_data["TestTargetsPerCommit"] >= 6]['RRF'].dropna()
        MART_result = rl_bs2_data[(rl_bs2_data["MART"] != 0) & (rl_bs2_data["TestTargetsPerCommit"] >= 6)]['MART'].dropna()
        ttest_rl_bs2, effect_size_rl_bs2 = calc_ttest(pairwise_ACER_result, rl_bs2_result)
        ttest_MART, effect_size_MART = calc_ttest(pairwise_ACER_result, MART_result)
        avg_diff.append(pairwise_ACER_result.mean() - rl_bs2_result.mean())
        avg_diff2.append(pairwise_ACER_result.mean() - MART_result.mean())
        print_rl_bs1 = f"RLBS2: {dataset} mean and STD: \small {rl_bs2_result.mean(skipna=True):.3f}$\pm${rl_bs2_result.std(skipna=True):.3f}" \
                 f" pvalue and CLE: \\small {ttest_rl_bs2.pvalue:.4f} & \\small {effect_size_rl_bs2:.3f}"
        print_MART = f"MART: {dataset} mean and std: \\small {MART_result.mean(skipna=True):.3f}$\pm${MART_result.std(skipna=True):.3f}" \
                 f" pvalue and CLE:  \\small {ttest_MART.pvalue:.4f} &  \\small {effect_size_MART:.3f}"
        print(print_rl_bs1)
        print(print_MART)
    print(mean(avg_diff))
    print(mean(avg_diff2))

    ## Taining time
    ## PAIRWISE-ACER training times
    datasets = {"simple":   ['iofrol-additional-features', 'paintcontrol-additional-features'],
                "enriched": ["Commons_codec", "Commons_imaging", "Commons_io",
                            "Commons_compress","Commons_lang","Commons_math"]}
    pairwise_ACER_training= {}
    pairwise_ACER_nrpa={}
    for dataset_type in datasets.keys():
        for dataset in datasets[dataset_type]:
            pairwise_ACER_training[ds_print[dataset]] = []
            pairwise_ACER_training[ds_print[dataset]] = \
                training_time_dataset_result[dataset]["PAIRWISE-ACER"]
            if dataset_type == "enriched":
                pairwise_ACER_nrpa[ds_print[dataset]] = []
                pairwise_ACER_nrpa[ds_print[dataset]] = \
                    accuracy_dataset_result[dataset]["PAIRWISE-ACER"]

    data = pd.DataFrame(pairwise_ACER_training)
    data.melt().to_csv("../experiments/acer-pairwise-training-time.csv")
    data1 = pd.DataFrame(pairwise_ACER_nrpa)
    data1.to_csv("../experiments/acer-pairwise-nrpa.csv")
    ## accuracy per cycle




