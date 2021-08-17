import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import csv

class TradingEnv(gym.Env):

    def __init__(self):
        self.n_stocks = 10
        self.W = 3
        self.count = 0
        self.max_steps = None
        self.action = [1]+[1/self.n_stocks]*self.n_stocks
        self.state = None

        ## spaces
        self.action_space = spaces.Box(low=0, high=1.0, shape=(self.n_stocks+1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(self.n_stocks+1, self.W), dtype=np.float32)
        

    def seed(self, seed=None):
        pass


    def reset(self):
        self.count = 0
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

    def calculate_reward(self):
        reward = np.log(self.beta*np.dot(self.state[0], self.action))
        done = False
        if(self.count>=self.max_steps):
            done = True
        return reward, done
        #valueOfHolding = data["Close"]
        #self.portifolio = valueOfHolding*self.holdings
        



    def step(self, action, data):
        state = self.receive_state(action, data)
        #print(state)
        self.count +=1

        reward, done = self.calculate_reward()
        #self.history.insert(0, [self.count, state, reward])
        #if(len(self.history)>3):
        #    self.history.pop(3)
        #print(self.history[0][1])

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

        return observation, step_reward, self._done, []


   
    def readData(self):
      ficheiro = open('gym_anytrading/datasets/data/STOCKS_AMBEV.csv', 'r')
      reader = csv.DictReader(ficheiro, delimiter = ',')
      #print(reader)
      #for linha in reader:
      #    print (linha["Close"])
      return reader