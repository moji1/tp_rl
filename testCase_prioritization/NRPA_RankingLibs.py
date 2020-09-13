import pandas as pd
import statistics

def calc_nrpa(cycle_data):
    return 0

def calc_score_ranking(ranks: list):
    if not ranks:
        return 0
    elif len(ranks) <= 1:
        return ranks[0]+1
    else:
        return (ranks[0]+1)*len(ranks) + calc_score_ranking(ranks[1:])

def get_optimal_RPA(n:int):
    if n==1:
        return 1
    else:
        return (n*n) + get_optimal_RPA(n-1)

if __name__ == '__main__':
    results = pd.read_csv(r"../data/langScore_1_ranker0.txt", sep='\t',
                      names = ["cycle_id", "rank", "assigned_rank"])
    cycle_ids = results["cycle_id"].unique()
    nrpas = []
    for cycle_id in cycle_ids:
        cycle_data = results[results["cycle_id"] == cycle_id]
        sorted_cycle = cycle_data.sort_values(by=["assigned_rank"], ascending=True, inplace=False)
        if len(sorted_cycle["rank"].tolist())>=6:
            rpa_order = calc_score_ranking(sorted_cycle["rank"].tolist())
            rpa_optimal = get_optimal_RPA(len(sorted_cycle["rank"].tolist()))
            nrpa = rpa_order/rpa_optimal
            nrpas.append(nrpa)
    print(statistics.mean(nrpas))


