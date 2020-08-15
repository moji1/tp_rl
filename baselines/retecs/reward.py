import numpy as np


def simple_discrete_reward(result, sc=None):
    if result[1] > 0:
        scenario_reward = -1.0
    elif result[0] > 0:
        scenario_reward = 1.0
    else:
        scenario_reward = 0.0

    return scenario_reward


def simple_continuous_reward(result, sc=None):
    failure_count = result[0] + result[1]

    if result[1] > 0:
        scenario_reward = -1.0  #- (3*result[1]/failure_count)
    elif result[0] > 0:
        scenario_reward = 1.0 + (3.0-3.0*result[2])
    else:
        scenario_reward = 0.0

    return scenario_reward


def napfd_reward(result, sc=None):
    total_failures = result[0] + result[1]
    scaling_factor = 1.0

    if total_failures == 0:
        return 0.0
    elif result[0] == 0:
        return -1 * scaling_factor
    else:
        # Apply NAPFD
        return result[3] * scaling_factor


def shifted_napfd_reward(result, sc=None):
    total_failures = result[0] + result[1]
    scaling_factor = 1.0

    if total_failures == 0:
        return 0.0
    elif result[0] == 0:
        return -1.0 * scaling_factor
    elif result[3] < 0.3:
        return result[3]-0.3
    else:
        # Apply NAPFD
        return result[3] * scaling_factor


def binary_positive_detection_reward(result, sc=None):
    rew = 1 if result[0] > 0 else 0
    return float(rew)


def failcount(result, sc=None):
    return float(result[0])


def timerank(result, sc):
    if result[0] == 0:
        return 0.0

    total = result[0]
    rank_idx = np.array(result[-1])-1
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1
    rewards = np.cumsum(rewards)  # Rewards for passed testcases
    rewards[rank_idx] = total  # Rewards for failed testcases

    ordered_rewards = []

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)  # Slow call
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    return ordered_rewards


def tcfail(result, sc):
    if result[0] == 0:
        return 0.0

    total = result[0]
    rank_idx = np.array(result[-1])-1
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1

    ordered_rewards = []

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    return ordered_rewards
