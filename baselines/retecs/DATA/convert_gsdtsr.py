#!/usr/bin/env python2
# Reads test suite information from google-shared-dataset-test-suite-results
# and creates a csv file usable by RETECS.
from __future__ import division, print_function
from datetime import datetime, timedelta
import csv
import numpy as np
import pandas as pd
import os
import sys
import gzip


def convert_gsdtsr(filename):
    basedate = datetime(year=2016, month=1, day=1)

    def launch2date(launch):
        time_parts = [int(x) for x in launch.split(':')]
        delta = timedelta(days=time_parts[0], hours=time_parts[1], minutes=time_parts[2], seconds=time_parts[3])
        return basedate + delta

    print('Read raw GSDTSR data...')
    with gzip.open(filename, 'rb') as f:
        df = pd.read_csv(f,
                         delimiter=',',
                         header=None,
                         names=['name', 'chgreq', 'stage', 'status', 'launch', 'duration', 'size', 'shard', 'run',
                                'language'],
                         usecols=[0, 1, 2, 3, 4, 5, 7],
                         true_values=['FAILED'],
                         false_values=['PASSED'])

    df['launch'] = df['launch'].apply(launch2date)
    df = df.sort_values(by=['launch'])
    df['name'] = pd.factorize(df.name)[0] + 1

    reddf = df.groupby(['name', 'chgreq', 'stage'], as_index=False).agg(
        {'status': any, 'launch': np.min, 'duration': np.sum})

    tc_fieldnames = ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults', 'Verdict', 'Cycle']

    tcdf = pd.DataFrame(columns=tc_fieldnames, index=reddf.index)
    tcdf['Id'] = reddf.index + 1
    tcdf['Name'] = reddf['name']
    tcdf['Duration'] = reddf['duration']
    tcdf['CalcPrio'] = 0
    tcdf['LastRun'] = reddf['launch']
    tcdf['Verdict'] = reddf['status'].apply(lambda x: 1 if x else 0)

    tcdf = tcdf.sort_values(by='LastRun')

    print('Collect historical test results (this takes some time)...')
    no_tcs = len(tcdf.Name.unique())
    for tccount, name in enumerate(tcdf.Name.unique(), start=1):
        verdicts = tcdf.loc[tcdf['Name'] == name, 'Verdict'].tolist()

        if len(verdicts) > 1:
            tcdf.loc[tcdf['Name'] == name, 'LastResults'] = [None] + [verdicts[i::-1] for i in
                                                                      range(0, len(verdicts) - 1)]
        sys.stdout.write('\r%.2f%%' % (tccount / no_tcs * 100))
        sys.stdout.flush()

    print('... done')
    tcdf['monthdayhour'] = tcdf['LastRun'].apply(lambda x: (x.month, x.day, x.hour))
    tcdf['Cycle'] = pd.factorize(tcdf.monthdayhour)[0] + 1
    #del tcdf['Name']
    del tcdf['monthdayhour']

    # tcdf.to_pickle('gsdtsr.p')  # Store as pickle for faster access through pandas
    print('Store results in gsdtsr.csv')
    tcdf.to_csv('gsdtsr.csv', sep=';', na_rep='[]', columns=tc_fieldnames, header=True, index=False,
                quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    convert_gsdtsr('testShareData.csv.rev.gz')
