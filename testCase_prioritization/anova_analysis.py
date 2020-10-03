from pingouin import welch_anova, read_dataset
import pingouin as pg
from sqlalchemy import create_engine
import pymysql
import pandas as pd
from scipy import stats
import math
import scipy.stats as stats


def calc_wilcoxon(sql_text1, sql_text2, result_table):
    sqlEngine = create_engine('mysql+pymysql://xxx127.0.0.1', pool_recycle=3600)
    dbConnection = sqlEngine.connect()
    data1 = pd.read_sql(sql_text1, dbConnection)
    data2 = pd.read_sql(sql_text2, dbConnection)
    '''if len(data1["var"]) == len(data2["var"]):
        wilcoxon_result = pg.wilcoxon(data1["var"], data2["var"], tail='two-sided')
    else:
        wilcoxon_result = pg.mwu(data1["var"], data2["var"], tail='two-sided')
    '''
    ttest = stats.ttest_ind(data1["var"], data2["var"], equal_var=False)
    effect_size = pg.compute_effsize(data1["var"],
                                     data2["var"], eftype='CLES')
    return  ttest, effect_size
    #wilcoxon_result.to_sql(result_table, dbConnection, schema=db_name, if_exists='replace')
    dbConnection.close()

def calc_welch_anova_generic(db_name, table_name, sql_text, annova_table,effect_table):
    sqlEngine = create_engine('mysql+pymysql://root:***!@#@127.0.0.1', pool_recycle=3600)
    dbConnection = sqlEngine.connect()
    data = pd.read_sql(sql_text,dbConnection)
    aov = welch_anova(dv='var', between='config', data=data)
    print(aov)
    pairwise_data = pg.pairwise_gameshowell(data=data, dv='var', between='config').round(3)
    gt_relation = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] > 0.0)]["A"]
    lt_relation = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] < 0)]["B"]
    all_significant = pd.DataFrame(pd.concat([gt_relation, lt_relation]), columns=["config"])
    best_config = all_significant.describe().loc["top"][0]
    all_config = pd.DataFrame(pd.concat([pairwise_data["A"], pairwise_data["B"]]), columns=["config"])
    worst_config = all_config[~all_config.config.isin(all_significant["config"])].iloc[0][0]
    if isinstance(best_config, str) and isinstance(worst_config, str):
        effect_size = pg.compute_effsize(data[data["config"] == best_config]["var"],
                       data[data["config"] == worst_config]["var"], eftype='CLES')
    all_config = all_config["config"].unique()
    effect_sizes=[]
    for config_m in all_config:
        for config_s in all_config:
            effect_size = pg.compute_effsize(data[data["config"] == config_m]["var"],
                                             data[data["config"] == config_s]["var"], eftype='CLES')
            rec = {"config1": config_m, "config2": config_s, "effect_size": effect_size}
            effect_sizes.append(rec)
    effect_sizes_df = pd.DataFrame(effect_sizes)
    #print(pairwise_data)
    pairwise_data.to_sql(annova_table, dbConnection, schema=db_name, if_exists='replace')
    effect_sizes_df.to_sql(effect_table, dbConnection, schema=db_name, if_exists='replace');
    dbConnection.close()



def calc_welch_anova_accuracy(db_name, table_name,nrpa_check, annova_table,effect_table):
    sqlEngine = create_engine('mysql+pymysql://root:***!@#@127.0.0.1', pool_recycle=3600)
    dbConnection = sqlEngine.connect()
    if nrpa_check:
        sql_text=f"select '{table_name}' ,concat(`mode`,'_',algo) as config, nrpa as var FROM {db_name}.{table_name}"
    else:
        sql_text = f"select '{table_name}' ,concat(`mode`,'_',algo) as config, apfd/optimal_apfd as var " \
                   f"FROM {db_name}.{table_name} where apfd>0"
    data = pd.read_sql(sql_text,dbConnection)

    aov = welch_anova(dv='var', between='config', data=data)
    print(aov)
    pairwise_data = pg.pairwise_gameshowell(data=data, dv='var', between='config').round(3)
    gt_relation = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] > 0.0)]["A"]
    lt_relation = pairwise_data[(pairwise_data["pval"] <= 0.05) & (pairwise_data["diff"] < 0)]["B"]
    all_significant = pd.DataFrame(pd.concat([gt_relation, lt_relation]), columns=["config"])
    best_config = all_significant.describe().loc["top"][0]
    all_config = pd.DataFrame(pd.concat([pairwise_data["A"], pairwise_data["B"]]), columns=["config"])
    worst_config = all_config[~all_config.config.isin(all_significant["config"])].iloc[0][0]
    if isinstance(best_config, str) and isinstance(worst_config, str):
        effect_size = pg.compute_effsize(data[data["config"] == best_config]["var"],
                       data[data["config"] == worst_config]["var"], eftype='CLES')
    all_config = all_config["config"].unique()
    effect_sizes=[]
    for config_m in all_config:
        for config_s in all_config:
            effect_size = pg.compute_effsize(data[data["config"] == config_m]["var"],
                                             data[data["config"] == config_s]["var"], eftype='CLES')
            rec = {"config1": config_m, "config2": config_s, "effect_size": effect_size}
            effect_sizes.append(rec)
    effect_sizes_df = pd.DataFrame(effect_sizes)
    print(pairwise_data)
    pairwise_data.to_sql(annova_table, dbConnection, schema=db_name, if_exists='replace')
    effect_sizes_df.to_sql(effect_table, dbConnection, schema=db_name, if_exists='replace');
    dbConnection.close()

def calc_welch_anova1(db_name,sql_text, result_table):
    sqlEngine = create_engine('mysql+pymysql://root:**!@#@127.0.0.1', pool_recycle=3600)
    dbConnection = sqlEngine.connect()
    data = pd.read_sql(sql_text,dbConnection)

    #df = read_dataset('anova')
    aov = welch_anova(dv='var', between='config', data=data)
    print(aov)
    tt = pg.pairwise_gameshowell(data=data, dv='var', between='config').round(3)
    print(tt)
    tt.to_sql(result_table,  dbConnection, schema=db_name, if_exists='replace');
    dbConnection.close()

if __name__ == '__main__':
    db_name="TPData"
    table_names = ["paintcontrol_1000_10","iofrol_1000_15","io_1000",
                 "codec_1000", "imaging_1000","compress_1000","lang_1000","math_1000"]
    nrpa_flags = {"paintcontrol_1000_10":False,"iofrol_1000_15":False,
                "io_1000":True, "codec_1000":True, "imaging_1000":True,
                "compress_1000":True, "lang_1000":True, "math_1000":True}
    shared_algo = ["A2C","TRPO","PPO1","PPO2"]
    ranking_models = ["PAIRWISE","POINTWISE","LISTWISE"]

    ## RQ1, anova anaysis of configurations
    '''≈ for table in table_names:
        calc_welch_anova_accuracy(db_name,table,nrpa_flags[table],"anova_"+table,"effect_"+table )
        sql_text_training = f"select '{table}' ,concat(`mode`,'_',algo) as " \
                            f"config, training_time / (1000 * 60) as var FROM {db_name}.{table}"
        calc_welch_anova_generic(db_name, table, sql_text_training, "anova_train_" + table,
                                 "effect_train_" + table)
    
    ## RQ1.2
    for algo in shared_algo:
        sql_text_enriched=f"select `mode` as config, nrpa as var from " \
                 f"( SELECT * FROM TPData.io_1000 union " \
                 f"SELECT * FROM TPData.codec_1000 union" \
                 f" SELECT * FROM TPData.compress_1000 union " \
                 f"SELECT * FROM TPData.math_1000 union" \
                 f"  SELECT * FROM TPData.imaging_1000 union  " \
                 f"SELECT * FROM TPData.lang_1000   ) temp " \
                 f"where algo='{algo}'"

        calc_welch_anova_generic(db_name, "", sql_text_enriched, "anova_accuracy_enriched_" + algo,
                                 "effect_accuracy_enriched_" + algo)


        sql_text_simple=f"select `mode` as config, apfd/optimal_apfd as var " \
                          f"from (SELECT * FROM TPData.paintcontrol_1000_10 where apfd>0 union " \
                          f"SELECT * FROM TPData.iofrol_1000_15 where apfd>0 ) temp " \
                          f"where algo='{algo}'"
        calc_welch_anova_generic(db_name, "", sql_text_simple, "anova_accuracy_simple_" + algo,
                                 "effect_accuracy_simple_" + algo)
        ## RQ1.3
        for mode in ranking_models:
            sql_text_accuracy_enriched = f"select `algo` as config, nrpa as var from " \
                                f"( SELECT * FROM TPData.io_1000 union " \
                                f"SELECT * FROM TPData.codec_1000 union" \
                                f" SELECT * FROM TPData.compress_1000 union " \
                                f"SELECT * FROM TPData.math_1000 union" \
                                f"  SELECT * FROM TPData.imaging_1000 union  " \
                                f"SELECT * FROM TPData.lang_1000   ) temp " \
                                f"where `mode`='{mode}'"

            calc_welch_anova_generic(db_name, "", sql_text_accuracy_enriched, "anova_accuracy_enriched_" + mode,
                                     "effect_accuracy_enriched_" + mode)

            sql_text_accuracy_simple = f"select `algo` as config, apfd/optimal_apfd as var " \
                              f"from (SELECT * FROM TPData.paintcontrol_1000_10 where apfd>0 union " \
                              f"SELECT * FROM TPData.iofrol_1000_15 where apfd>0 ) temp " \
                              f"where `mode`='{mode}'"
            calc_welch_anova_generic(db_name, "", sql_text_accuracy_simple, "anova_accuracy_simple_" + mode,
                                     "effect_accuracy_simple_" + mode)
            ############
            sql_text_trainig_time_enriched = f"select `algo` as config, (training_time / (1000 * 60))/test_cases as var from " \
                                f"( SELECT * FROM TPData.io_1000 union " \
                                f"SELECT * FROM TPData.codec_1000 union" \
                                f" SELECT * FROM TPData.compress_1000 union " \
                                f"SELECT * FROM TPData.math_1000 union" \
                                f"  SELECT * FROM TPData.imaging_1000 union  " \
                                f"SELECT * FROM TPData.lang_1000   ) temp " \
                                f"where `mode`='{mode}'"

            calc_welch_anova_generic(db_name, "", sql_text_trainig_time_enriched, "anova_train_time_enriched_" + mode,
                                     "effect_train_time_enriched_" + mode)

            sql_text_trainig_time_simple = f"select `algo` as config, (training_time / (1000 * 60))/test_cases as var " \
                                       f"from (SELECT * FROM TPData.paintcontrol_1000_10 where apfd>0 union " \
                                       f"SELECT * FROM TPData.iofrol_1000_15 where apfd>0 ) temp " \
                                       f"where `mode`='{mode}'"
            calc_welch_anova_generic(db_name, "", sql_text_trainig_time_simple, "anova_train_time_simple_" + mode,
                                     "effect_train_time_simple_" + mode)
    ## RQ1.2 anova anaysis of configurations of ranking models
    '''
    ## RQ2
    icse_paper_tables ={"io_1000":"io-normal-ICSE", "codec_1000":"codec-normal-icse",
                     "imaging_1000":"imaging-normal-ICSE","compress_1000":"compress-normal-ICSE",
                     "lang_1000":"lang-normal-icse","math_1000":"math-normal-icse"}
    issta_paper_tables = {"paintcontrol_1000_10":"paintcontrol_issta","iofrol_1000_15":"iofrol_issta"}
    for table in icse_paper_tables.keys():
        sql_text_our_result = f"select nrpa as var from {db_name}.{table} where" \
                              f" algo='ACER' and `mode`='PAIRWISE'"
        sql_text_icse_results_mart = f"SELECT mart as var FROM {db_name}.`{icse_paper_tables[table]}`" \
                                     f" where mart>0 and test_cases>=6"
        ttest1, cle1= calc_wilcoxon(sql_text_our_result, sql_text_icse_results_mart, "wilcoxon_icse_mart_"+table)
        sql_text_icse_results_rl = f"SELECT rrf as var FROM {db_name}.`{icse_paper_tables[table]}`" \
                                     f" where  test_cases>=6"
        ttest2, cle2= calc_wilcoxon(sql_text_our_result, sql_text_icse_results_rl, "wilcoxon_icse_rl_"+icse_paper_tables[table])
        logs=f"\small {table} & NA & NA &  \small {'{0: 1.4f}'.format(ttest2.pvalue)} " \
             f" & \small {'{0: 1.4f}'.format(cle2)}  & \small {'{0: 1.4f}'.format(ttest1.pvalue)} " \
             f"& \small {'{0: 1.3f}'.format(cle1)} \\\\"
        print(logs)
