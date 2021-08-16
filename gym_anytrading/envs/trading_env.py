import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import csv

class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        #assert df.ndim == 2
        #
        #self.seed()
        #self.df = df
        #self.window_size = window_size
        #self.prices, self.signal_features = self._process_data()
        #self.shape = (window_size, self.signal_features.shape[1])

        ## spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        self.holdings = 0
        self.balance = 0
        self.portifolio  = 10000

        ## episode
        #self._start_tick = self.window_size
        #self._end_tick = len(self.prices) - 1
        #self._done = None
        #self._current_tick = None
        #self._last_trade_tick = None
        #self._position = None
        #self._position_history = None
        #self._total_reward = None
        #self._total_profit = None
        #self._first_rendering = None
        #self.history = None
        pass

    def seed(self, seed=None):
        #self.np_random, seed = seeding.np_random(seed)
        #return [seed]
        pass


    def reset(self):
        #self._done = False
        #self._current_tick = self._start_tick
        #self._last_trade_tick = self._current_tick - 1
        #self._position = Positions.Short
        #self._position_history = (self.window_size * [None]) + [self._position]
        #self._total_reward = 0.
        #self._total_profit = 1.  # unit
        #self._first_rendering = True
        #self.history = {}
        #return self._get_observation()
        pass

    def receive_state(self, action, data):
        self.holdings = self.holdings - action

    def calculate_reward(self, data):
        valueOfHolding = data["Close"]
        self.portifolio = valueOfHolding*self.holdings
        



    def step(self, action, data):
        state = self.receive_state(action, data)
        #print(state)
        reward, done = self.compute_metrics(self.history,state)
        self.history.insert(0, [self.count, state, reward])
        if(len(self.history)>3):
            self.history.pop(3)
        #print(self.history[0][1])
        self.count +=1

        #self._done = False
        #self._current_tick += 1

        #if self._current_tick == self._end_tick:
        #    self._done = True

        #step_reward = self._calculate_reward(action)
        #self._total_reward += step_reward

        #self._update_profit(action)

        #trade = False
        #if ((action == Actions.Buy.value and self._position == Positions.Short) or
        #    (action == Actions.Sell.value and self._position == Positions.Long)):
        #    trade = True

        #if trade:
        #    self._position = self._position.opposite()
        #    self._last_trade_tick = self._current_tick

        #self._position_history.append(self._position)
        #observation = self._get_observation()
        #info = dict(
        #    total_reward = self._total_reward,
        #    total_profit = self._total_profit,
        #    position = self._position.value
        #)
        #self._update_history(info)

        #return observation, step_reward, self._done, info


   
    def readData(self):
      ficheiro = open('gym_anytrading/datasets/data/STOCKS_AMBEV.csv', 'r')
      reader = csv.DictReader(ficheiro, delimiter = ',')
      #print(reader)
      #for linha in reader:
      #    print (linha["Close"])
      return reader