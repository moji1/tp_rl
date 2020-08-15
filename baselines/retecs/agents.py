import numpy as np
import os
from sklearn import neural_network

try:
    import cPickle as pickle
except:
    import pickle


class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, experience):
        self.memory.append(experience)

    def get_batch(self, batch_size=10):
        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
            batch_idx = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=p)
            batch = [self.memory[idx] for idx in batch_idx]
        else:
            batch = self.memory

        return batch


class BaseAgent(object):
    def __init__(self, histlen):
        self.single_testcases = True
        self.train_mode = True
        self.histlen = histlen

    def get_action(self, s):
        return 0

    def get_all_actions(self, states):
        """ Returns list of actions for all states """
        return [self.get_action(s) for s in states]

    def reward(self, reward):
        pass

    def save(self, filename):
        """ Stores agent as pickled file """
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


class TableauAgent(BaseAgent):
    def __init__(self, learning_rate, state_size, action_size, epsilon, histlen):
        # Key: (State Representation) -> (N, Q)
        super(TableauAgent, self).__init__(histlen=histlen)
        self.name = 'tableau'
        self.state_in = state_size
        self.states = {}  # The Tableau
        self.initial_q = 5
        self.action_history = []
        self.action_size = action_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_epsilon = epsilon
        self.min_epsilon = 0.1
        self.gamma = 0.99

    def get_action(self, s):
        if s not in self.states:
            self.states[s] = {
                'Q': [self.initial_q] * self.action_size,
                'N': [0] * self.action_size
            }

        if np.random.rand() >= self.epsilon:
            action = self.random_argmax(self.states[s]['Q'])
        else:
            action = np.random.randint(self.action_size)

        if self.train_mode:
            self.action_history.append((s, action))

        return action

    def reward(self, rewards):
        if not self.train_mode:
            return

        try:
            x = float(rewards)
            rewards = [x] * len(self.action_history)
        except:
            if len(rewards) < len(self.action_history):
                raise Exception('Too few rewards')

        # Update Q
        for ((state, act_idx), reward) in zip(self.action_history, rewards):
            self.states[state]['N'][act_idx] += 1
            n = self.states[state]['N'][act_idx]
            prev_q = self.states[state]['Q'][act_idx]
            self.states[state]['Q'][act_idx] = prev_q + 1.0 / n * (reward - prev_q)
            # self.states[state]['Q'][act_idx] = prev_q + self.learning_rate * (reward - prev_q)

        self.reset_action_history()
        self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon

    def reset_action_history(self):
        self.action_history = []

    @staticmethod
    def random_argmax(vector):
        """ Argmax that chooses randomly among eligible maximum indices. """
        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]
        return np.random.choice(indices)


class NetworkAgent(BaseAgent):
    def __init__(self, state_size, action_size, hidden_size, histlen):
        super(NetworkAgent, self).__init__(histlen=histlen)
        self.name = 'mlpclassifier'
        self.experience_length = 10000
        self.experience_batch_size = 1000
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []
        self.iteration_counter = 0
        self.action_size = action_size

        if isinstance(hidden_size, tuple):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = (hidden_size,)
        self.model = None
        self.model_fit = False
        self.init_model(True)

    # TODO This could improve performance (if necessary)
    # def get_all_actions(self, states):
    #   try:

    def init_model(self, warm_start=True):
        if self.action_size == 1:
            self.model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                      warm_start=warm_start, solver='adam', max_iter=750)
        else:
            self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                     warm_start=warm_start, solver='adam', max_iter=750)
        self.model_fit = False

    def get_action(self, s):
        if self.model_fit:
            if self.action_size == 1:
                a = self.model.predict_proba(np.array(s).reshape(1, -1))[0][1]
            else:
                a = self.model.predict(np.array(s).reshape(1, -1))[0]
        else:
            a = np.random.random()

        if self.train_mode:
            self.episode_history.append((s, a))

        return a

    def reward(self, rewards):
        if not self.train_mode:
            return

        try:
            x = float(rewards)
            rewards = [x] * len(self.episode_history)
        except:
            if len(rewards) < len(self.episode_history):
                raise Exception('Too few rewards')

        self.iteration_counter += 1

        for ((state, action), reward) in zip(self.episode_history, rewards):
            self.experience.remember((state, reward))

        self.episode_history = []

        if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            self.learn_from_experience()

    def learn_from_experience(self):
        experiences = self.experience.get_batch(self.experience_batch_size)
        x, y = zip(*experiences)

        if self.model_fit:
            try:
                self.model.partial_fit(x, y)
            except ValueError:
                self.init_model(warm_start=False)
                self.model.fit(x, y)
                self.model_fit = True
        else:
            self.model.fit(x, y)  # Call fit once to learn classes
            self.model_fit = True


class RandomAgent(BaseAgent):
    def __init__(self, histlen):
        super(RandomAgent, self).__init__(histlen=histlen)
        self.name = 'random'

    def get_action(self, s):
        return np.random.random()

    def get_all_actions(self, states):
        prio = range(len(states))
        np.random.shuffle(prio)
        return prio


class HeuristicSortAgent(BaseAgent):
    """ Sort first by last execution results, then time not executed """

    def __init__(self, histlen):
        super(HeuristicSortAgent, self).__init__(histlen=histlen)
        self.name = 'heuristic_sort'
        self.single_testcases = False

    def get_action(self, s):
        raise NotImplementedError('Single get_action not implemented for HeuristicSortAgent')

    def get_all_actions(self, states):
        sorted_idx = sorted(range(len(states)),
                            key=lambda x: list(states[x][-self.histlen:]) + [states[x][-self.histlen - 1]])
        sorted_actions = sorted(range(len(states)), key=lambda i: sorted_idx[i])
        return sorted_actions


class HeuristicWeightAgent(BaseAgent):
    """ Sort by weighted representation """

    def __init__(self, histlen):
        super(HeuristicWeightAgent, self).__init__(histlen=histlen)
        self.name = 'heuristic_weight'
        self.single_testcases = False
        self.weights = []

    def get_action(self, s):
        raise NotImplementedError('Single get_action not implemented for HeuristicWeightAgent')

    def get_all_actions(self, states):
        if len(self.weights) == 0:
            state_size = len(states[0])
            self.weights = np.ones(state_size) / state_size

        sorted_idx = sorted(range(len(states)), key=lambda x: sum(states[x] * self.weights))
        sorted_actions = sorted(range(len(states)), key=lambda i: sorted_idx[i])
        return sorted_actions


def restore_agent(model_file):
    if os.path.exists(model_file + '.p'):
        return BaseAgent.load(model_file)
    else:
        raise Exception('Not a valid agent')
