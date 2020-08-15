from __future__ import division
import json
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import random
import copy

try:
    import cPickle as pickle
except:
    import pickle


def inhomogeneous_poisson(l, rej_threshold, default=0, size=1):
    values = np.random.poisson(lam=l, size=1)
    rnd_throws = np.random.uniform(size=values.shape)
    values[rnd_throws < rej_threshold] = default
    return values


def generate_testcase(id, last_run, duration_limits=[180, 1200], history_length=0, history_fail_prob=0.05):
    tc = {
        'Id': id,
        'Duration': random.randint(duration_limits[0], duration_limits[1]),
        'CalcPrio': 0,
        'LastRun': last_run,
        'LastResults': [1 if random.random() < history_fail_prob else 0 for _ in range(history_length)]
    }
    return tc


def generate_solution(tc, basic_failure_chance, prev_failure_influence):
    failure_chance = basic_failure_chance + sum(tc['LastResults'][0:3]) * prev_failure_influence
    return 1 if random.random() < failure_chance else 0


class VirtualScenario(object):
    def __init__(self, available_time, testcases=[], solutions={}, name_suffix='vrt', schedule_date=datetime.today()):
        self.available_time = available_time
        self.gen_testcases = testcases
        self.solutions = solutions
        self.no_testcases = len(testcases)
        self.name = name_suffix
        self.scheduled_testcases = []
        self.schedule_date = schedule_date

    def testcases(self):
        return iter(self.gen_testcases)

    def submit(self):
        # Sort tc by Prio ASC (for backwards scheduling), break ties randomly
        sorted_tc = sorted(self.gen_testcases, key=lambda x: (x['CalcPrio'], random.random()))
        order = sorted_tc.copy()
        # print(sorted_tc)
        # Build prefix sum of durations to find cut off point
        scheduled_time = 0
        detection_ranks = []
        detection_ranks_all = []
        undetected_failures = 0
        rank_counter = 1
        sorted_tc1 = sorted_tc.copy()
        while sorted_tc1:
            cur_tc = sorted_tc1.pop()
            if self.solutions[cur_tc['Id']]:
                detection_ranks_all.append(rank_counter)
            rank_counter += 1

        rank_counter = 1
        while sorted_tc:
            cur_tc = sorted_tc.pop()

            if scheduled_time + cur_tc['Duration'] <= self.available_time:
                if self.solutions[cur_tc['Id']]:
                    detection_ranks.append(rank_counter)

                scheduled_time += cur_tc['Duration']
                self.scheduled_testcases.append(cur_tc)
                rank_counter += 1
            else:
                undetected_failures += self.solutions[cur_tc['Id']]

        detected_failures = len(detection_ranks)
        detected_failures_all = len(detection_ranks_all)
        total_failure_count = sum(self.solutions.values())

        assert undetected_failures + detected_failures == total_failure_count

        if total_failure_count > 0:
            ttf = detection_ranks[0] if detection_ranks else 0

            if undetected_failures > 0:
                p = (detected_failures / total_failure_count)
            else:
                p = 1

            napfd = p - sum(detection_ranks) / (total_failure_count * self.no_testcases) + p / (2 * self.no_testcases)
            apfd  = 1 - sum(detection_ranks_all) / (total_failure_count * self.no_testcases) + 1 / (2 * self.no_testcases)
            recall = detected_failures / total_failure_count
            avg_precision = 123
        else:
            ttf = 0
            napfd = 0
            apfd= 0
            recall = 0
            avg_precision = 0

        return [detected_failures, undetected_failures, ttf, apfd, napfd, recall, avg_precision, order, detection_ranks]

    def get_ta_metadata(self):
        execTimes, durations = zip(*[(tc['LastRun'], tc['Duration']) for tc in self.testcases()])

        metadata = {
            'availAgents': 1,
            'totalTime': self.available_time,
            'minExecTime': min(execTimes),
            'maxExecTime': max(execTimes),
            'scheduleDate': self.schedule_date,
            'minDuration': min(durations),
            'maxDuration': max(durations)
        }

        return metadata

    def set_testcase_prio(self, prio, tcid=-1):
        self.gen_testcases[tcid]['CalcPrio'] = prio

    def reduce_to_schedule(self):
        """ Creates a new scenario consisting of all scheduled test cases and their outcomes (for replaying) """
        scheduled_time = sum([tc['Duration'] for tc in self.scheduled_testcases])
        total_time = sum([tc['Duration'] for tc in self.testcases()])
        available_time = self.available_time * scheduled_time / total_time
        solutions = {tc['Id']: self.solutions[tc['Id']] for tc in self.scheduled_testcases}
        return VirtualScenario(available_time, self.scheduled_testcases, solutions, self.name, self.schedule_date)

    def clean(self):
        for tc in self.testcases():
            self.set_testcase_prio(0, tc['Id'] - 1)

        self.scheduled_testcases = []


class RandomScenario(VirtualScenario):
    """ On-the-fly random scenario generator for schedules with only one test agent and without schedule optimization"""

    def __init__(self, schedule_ratio=None, no_testcases=None, history_length=3, init_testcases=False,
                 name_suffix='rnd'):
        super(RandomScenario, self).__init__(available_time=random.randint(14400, 28800), name_suffix=name_suffix)
        self.tc_duration_limit = [180, 1200]
        self.must_run_prob = 0.2
        self.basic_failure_chance = 0.03
        self.prev_failure_influence = 0.5
        self.history_length = history_length

        if no_testcases is None:
            time_to_schedule = self.available_time / schedule_ratio
            self.no_testcases = int(time_to_schedule / np.mean(self.tc_duration_limit))
            self.name = '1_%.1f_%s' % (schedule_ratio, name_suffix)
        else:
            self.no_testcases = no_testcases
            self.name = '1_%d_%s' % (no_testcases, name_suffix)

        self.gen_testcases = []
        self.scheduled_testcases = []
        self.solutions = {}

        if init_testcases:
            list(self.testcases())

    def testcases(self):
        if len(self.gen_testcases) < self.no_testcases:
            for i in range(len(self.gen_testcases), self.no_testcases):
                yield self.generate_testcase()
        else:
            for i in range(self.no_testcases):
                yield self.gen_testcases[i]

    def generate_testcase(self):
        last_run = self.schedule_date - timedelta(days=random.randint(1, 5))

        tc = generate_testcase(id=len(self.gen_testcases) + 1, duration_limits=self.tc_duration_limit,
                               last_run=last_run, history_length=self.history_length)

        self.gen_testcases.append(tc)

        sol = self.generate_solution(tc)
        self.solutions[tc['Id']] = sol
        return tc

    def generate_solution(self, tc):
        return generate_solution(tc, self.basic_failure_chance, self.prev_failure_influence)

    def clean(self):
        for tc in self.testcases():
            self.set_testcase_prio(0, tc['Id'] - 1)

        self.scheduled_testcases = []


class RandomScenarioProvider(object):
    def __init__(self, scenario_class=RandomScenario):
        self.schedule_ratios = [0.3, 0.5, 0.7, 0.9]
        self.validation = []
        self.validation_length = 64
        self.scenario_class = scenario_class
        self.name = 'random'

    def get(self, name_suffix='rnd', init_testcases=False):
        schedule_ratio = random.choice(self.schedule_ratios)
        return self.scenario_class(schedule_ratio=schedule_ratio, init_testcases=init_testcases,
                                   name_suffix=name_suffix)

    def get_validation(self):
        if not self.validation:
            if os.path.exists('%s_validation.p' % type(self).__name__):
                self.validation = pickle.load(open('%s_validation.p' % type(self).__name__, 'rb'))
            else:
                self.validation = [self.get(name_suffix='rnd%d' % i) for i in range(self.validation_length)]
                pickle.dump(self.validation, open('%s_validation.p' % type(self).__name__, 'wb'), 2)

        return copy.deepcopy(self.validation)

    # Generator functions
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


class IncrementalScenarioProvider(RandomScenarioProvider):
    def __init__(self, testcases=[], solutions={}, episode_length=50, avg_failure_cnt=5, prob_tc_changes=0.1):
        super(IncrementalScenarioProvider, self).__init__()
        self.name = 'incremental'
        self.episode_length = episode_length
        self.step_counter = 0
        self.scenario = None
        self.validation_length = self.episode_length * 2
        self.initial_last_run = datetime(2015, 1, 1)
        self.basic_failure_chance = 0.03
        self.prev_failure_influence = 0.15
        self.avg_failure_count = avg_failure_cnt
        self.prob_failure_count_changes = 0.9
        self.prob_tc_changes = prob_tc_changes
        self.prob_tc_add = 0.7
        self.testcases = testcases
        self.solutions = solutions
        self.available_time = 0

        if len(self.testcases) > 0 and len(self.solutions) > 0:
            self.scenario = VirtualScenario(testcases=self.testcases, solutions=self.solutions, name_suffix='inc0')

    def get_validation(self):
        if not self.validation:
            if os.path.exists('%s_validation.p' % type(self).__name__) and False:
                self.validation = pickle.load(open('%s_validation.p' % type(self).__name__, 'rb'))
            else:
                self.validation = [RandomScenario(no_testcases=100, name_suffix='rnd%d' % i) for i in
                                   range(self.validation_length)]
                pickle.dump(self.validation, open('%s_validation.p' % type(self).__name__, 'wb'), 2)

        return copy.deepcopy(self.validation)

    def get(self, name_suffix='inc'):
        if self.scenario is None or self.step_counter % self.episode_length == 0:
            self.scenario = super(IncrementalScenarioProvider, self).get(
                name_suffix='%s%d' % (name_suffix, self.step_counter))
            self.testcases = list(self.scenario.testcases())
            self.solutions = self.scenario.solutions
            self.available_time = self.scenario.available_time
        else:
            self.scenario = self.updated_scenario()

        self.step_counter += 1

        return self.scenario

    def updated_scenario(self):
        today = datetime.today()

        # Expected variation in failures
        if np.random.random() < self.prob_failure_count_changes:
            failure_count_changes = inhomogeneous_poisson(self.avg_failure_count, 1) - self.avg_failure_count
        else:
            failure_count_changes = 0

        # Update recently executed testcases
        for (idx, tc) in enumerate(self.testcases):
            if tc in self.scenario.scheduled_testcases:
                sol = self.solutions[tc['Id']]
                tc['LastResults'] = [sol] + tc['LastResults']
                tc['LastRun'] = today - timedelta(days=1)
                self.solutions[tc['Id']] = generate_solution(tc, self.basic_failure_chance, self.prev_failure_influence)
            else:
                tc['LastRun'] = tc['LastRun'] - timedelta(days=1)

                if random.random() < self.basic_failure_chance:
                    self.solutions[tc['Id']] = not self.solutions[tc['Id']]
                    failure_count_changes += -1 if self.solutions[tc['Id']] else +1

            self.testcases[idx] = tc

        # Update total testcase repository
        tc_changes = inhomogeneous_poisson(10, self.prob_tc_changes) - 10

        for i in range(tc_changes):
            if np.random.random() < self.prob_tc_add:
                # Add test case
                tc_id = max(self.solutions.keys()) + 1
                tc = generate_testcase(id=tc_id, last_run=self.initial_last_run)
                self.testcases.append(tc)
                sol = generate_solution(tc, self.basic_failure_chance, self.prev_failure_influence)
                self.solutions[tc_id] = sol

                if sol:
                    failure_count_changes -= 1
            else:
                # Remove random test case
                idx = np.random.randint(0, len(self.testcases))
                del self.solutions[self.testcases[idx]['Id']]
                del self.testcases[idx]

        if failure_count_changes != 0:
            if failure_count_changes > 0:
                cand_tc = [tc for tc in self.testcases if not self.solutions[tc['Id']]]
            else:
                cand_tc = [tc for tc in self.testcases if self.solutions[tc['Id']]]

            if len(cand_tc) >= abs(failure_count_changes):
                chg_tc = np.random.choice(cand_tc, size=abs(failure_count_changes))

                for tc in chg_tc:
                    self.solutions[tc['Id']] = not self.solutions[tc['Id']]

        assert len(self.testcases) == len(self.solutions)
        assert len([tc for tc in self.testcases if not tc['Id'] in self.solutions]) == 0

        name = 'inc%d' % self.step_counter
        return VirtualScenario(self.available_time, self.testcases, self.solutions, name_suffix=name)


class FileBasedSubsetScenarioProvider(RandomScenarioProvider):
    def __init__(self, tcfile, solfile, scheduleperiod=20, starttime=None, sched_time_ratio=0.5):
        super(FileBasedSubsetScenarioProvider, self).__init__()

        self.basename = os.path.splitext(os.path.basename(tcfile))[0]
        self.name = self.basename
        self.testcases = []
        self.solutions = {}

        self.tc_reader = csv.DictReader(open(tcfile, 'r'), delimiter=';', quoting=csv.QUOTE_MINIMAL, escapechar='',
                                        quotechar='\'')
        self.sol_reader = csv.DictReader(open(solfile, 'r'), delimiter=';', quoting=csv.QUOTE_MINIMAL, escapechar='',
                                         quotechar='\'')

        tc = self.next_testcase().next()
        self.testcases.append(tc)

        if starttime is None or not isinstance(starttime, datetime):
            self.starttime = tc['LastRun'].replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            self.starttime = starttime

        self.lastidx = -1
        self.maxtime = self.starttime
        self.scheduleperiod = scheduleperiod
        self.scenario = None
        self.avail_time_ratio = sched_time_ratio

    def next_testcase(self):
        for row in self.tc_reader:
            tc = self.row_to_testcase(row)
            self.load_solution(tc['Id'])  # Assure solution is loaded
            assert tc['Id'] in self.solutions
            yield tc

    def load_solution(self, tc_id):
        if tc_id in self.solutions:
            return

        for s in self.sol_reader:
            s_id = int(s['Id'])
            self.solutions[s_id] = s['Result'] == '1'

            if s_id == tc_id:
                break

    def row_to_testcase(self, tc):
        tc['Id'] = int(tc['Id'])
        tc['MustRun'] = tc['MustRun'] == '1'
        tc['Duration'] = int(tc['Duration'])
        tc['FixedPrio'] = int(tc['FixedPrio'])
        tc['LastResults'] = json.loads(tc['LastResults'])
        tc['LastAgents'] = [1] * len(tc['LastResults'])  # ast.literal_eval(tc['LastAgents'])
        tc['PossAgents'] = [1]  # ast.literal_eval(tc['PossAgents'])

        a = tc['LastRun']
        if len(tc['LastRun']) == 16:
            tc['LastRun'] = datetime(int(a[:4]), int(a[5:7]), int(a[8:10]), int(a[11:13]), int(a[14:16]))
        else:
            tc['LastRun'] = datetime(int(a[:4]), int(a[5:7]), int(a[8:10]), int(a[11:13]), int(a[14:16]), int(a[17:19]))

        return tc

    def get(self, name_suffix=None, init_testcases=False):
        seltc = self.testcases
        self.testcases = []

        if isinstance(self.scheduleperiod, timedelta):
            self.maxtime += self.scheduleperiod

        for tc in self.next_testcase():
            add_by_date = isinstance(self.scheduleperiod, timedelta) and tc['LastRun'] <= self.maxtime
            add_by_count = isinstance(self.scheduleperiod, int) and len(seltc) < self.scheduleperiod

            if add_by_date or add_by_count:
                seltc.append(tc)
            else:
                self.testcases.append(tc)
                break

        if len(seltc) > 0:
            if name_suffix is None:
                name_suffix = (self.maxtime + timedelta(days=1)).isoformat()

            req_time = sum([tc['Duration'] for tc in seltc])
            total_time = req_time * self.avail_time_ratio

            selsol = {tc['Id']: self.solutions[tc['Id']] for tc in seltc}
            self.scenario = VirtualScenario(testcases=seltc, solutions=selsol, name_suffix=name_suffix,
                                            available_time=total_time, schedule_date=self.maxtime + timedelta(days=1))
            self.maxtime = seltc[-1]['LastRun']
        else:
            self.scenario = None

        return self.scenario

    def get_validation(self):
        if not self.validation:
            val_path = '%s_%s_validation.p' % (type(self).__name__, self.basename)

            if os.path.exists(val_path):
                self.validation = pickle.load(open(val_path, 'rb'))
            else:
                self.validation = []

                while len(self.validation) < 14:
                    # Two periods of each 7 days
                    starttimes = sorted(
                        set([c['LastRun'].replace(hour=0, minute=0, second=0, microsecond=0) for c in self.testcases]))[
                                 :-7]

                    idx = random.randint(0, len(starttimes) - 1)
                    week = []
                    remove_tc = []

                    for j in range(7):
                        start = starttimes[idx + j]
                        end = starttimes[idx + j + 1]
                        seltc = [tc for tc in self.testcases if start < tc['LastRun'] <= end]
                        selsol = {tc['Id']: self.solutions[tc['Id']] for tc in seltc}
                        req_time = sum([tc['Duration'] for tc in seltc])
                        total_time = req_time * self.avail_time_ratio
                        val_scenario = VirtualScenario(testcases=seltc, solutions=selsol,
                                                       name_suffix='val_%s' % start.isoformat(),
                                                       available_time=total_time,
                                                       schedule_date=end + timedelta(days=1))

                        if sum(val_scenario.solutions.values()) == 0:
                            break  # Choose new starttime

                        week.append(val_scenario)
                        remove_tc.extend(val_scenario.testcases())
                    else:
                        self.validation.extend(week)
                        self.testcases[:] = [tc for tc in self.testcases if tc not in remove_tc]

                pickle.dump(self.validation, open(val_path, 'wb'), 2)

        return copy.deepcopy(self.validation)


class IndustrialDatasetScenarioProvider(RandomScenarioProvider):
    """
    Scenario provider to process CSV files for experimental evaluation of RETECS.

    Required columns are `self.tc_fieldnames` plus ['Verdict', 'Cycle']
    """
    def __init__(self, tcfile, sched_time_ratio=0.5):
        super(IndustrialDatasetScenarioProvider, self).__init__()

        self.basename = os.path.splitext(os.path.basename(tcfile))[0]
        self.name = self.basename

        self.tcdf = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'])
        self.tcdf['LastResults'] = self.tcdf['LastResults'].apply(json.loads)
        self.solutions = dict(zip(self.tcdf['Id'].tolist(), self.tcdf['Verdict'].tolist()))

        self.cycle = 0
        self.maxtime = min(self.tcdf.LastRun)
        self.max_cycles = max(self.tcdf.Cycle)
        self.scenario = None
        self.avail_time_ratio = sched_time_ratio
        self.tc_fieldnames = ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']

    def get(self, name_suffix=None):
        self.cycle += 1

        if self.cycle > self.max_cycles:
            self.scenario = None
            return None

        cycledf = self.tcdf.loc[self.tcdf.Cycle == self.cycle]

        seltc = cycledf[self.tc_fieldnames].to_dict(orient='record')

        if name_suffix is None:
            name_suffix = (self.maxtime + timedelta(days=1)).isoformat()

        req_time = sum([tc['Duration'] for tc in seltc])
        total_time = req_time * self.avail_time_ratio

        selsol = dict(zip(cycledf['Id'].tolist(), cycledf['Verdict'].tolist()))

        self.scenario = VirtualScenario(testcases=seltc, solutions=selsol, name_suffix=name_suffix,
                                        available_time=total_time, schedule_date=self.maxtime + timedelta(days=1))
        self.maxtime = seltc[-1]['LastRun']

        return self.scenario

    def get_validation(self):
        """ Validation data sets are not supported for this provider """
        return []


class ScenarioStore(object):
    def __init__(self, max_memory=500, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, scenario):
        self.memory.append(scenario)

        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, batch_size=10):
        batch = np.random.choice(self.memory, size=batch_size)
        return batch
