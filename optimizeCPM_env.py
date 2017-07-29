from random import random, randint

from tensorforce.environments import Environment
from scipy.stats import norm

FLOOR_PRICE = 100
N_EXCHANGES = 5
MAX_MEAN = FLOOR_PRICE / 2
MAX_VAR = FLOOR_PRICE / 20

class optimizeCPM(Environment):
    def __init__(self):
        self.hidden_state = [[randint(1, MAX_MEAN), randint(1, MAX_VAR)] for x in range(N_EXCHANGES)]

    def __str__(self):
        return 'optimizeCPM'

    def close(self):
        pass

    def reset(self):
        self.state = [0]
        self.count = 0
        return self.state

    def execute(self, action):
        exchange = action / FLOOR_PRICE
        floor = action % FLOOR_PRICE
        mean = self.hidden_state[exchange][0]
        var = self.hidden_state[exchange][1]
        if random() > norm.cdf(floor, mean, var):
            reward = action % FLOOR_PRICE + 0.5
        else:
            reward = 0
        terminal = True
        print action
        return self.state, reward, terminal

    @property
    def states(self):
        return dict(shape=(1,), type='float')

    @property
    def actions(self):
        return dict(continuous=False, num_actions=FLOOR_PRICE*N_EXCHANGES)