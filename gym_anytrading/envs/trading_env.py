import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import csv
import gym_anytrading.datasets.b3 as b3

class TradingEnv(gym.Env):

    def __init__(self):
        self.n_stocks = 10
        self.W = 2
        self.count = 0
        self.count_episodes = -1
        self.max_steps = 5
        #self.action = [1/(self.n_stocks+1)]*(self.n_stocks+1)
        self.state = None
        csv_filename = '../../../gym_anytrading/datasets/data/B3_COTAHIST.csv'
        #csv_filename = 'gym_anytrading/datasets/data/B3_COTAHIST.csv'
        self.df = pd.read_csv(csv_filename, parse_dates=True, index_col='Date')
        #print(self.df.head())


        ## spaces
        self.action_space = spaces.Box(low=0, high=1.0, shape=(self.n_stocks+1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=((self.W+1)*(self.n_stocks+1), ), dtype=np.float32)
        self.beta = 1

    def seed(self, seed=None):
        pass


    def reset(self):
        self.count = 0
        self.count_episodes += 1
        return self.receive_state().flatten()
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
        #pass
    def normalizeAction(self, action):
        new_action = []
        action = np.array(action)
        for i in action: #range(len(action)):
            new_action.append(i/action.sum())
        #print(new_action, np.array(new_action).sum())
        return new_action

    def receive_state(self):
        state = []
        #print("AQUI.......")
        for j in range(self.W, -1, -1):
            start_point =self.n_stocks*self.W + self.count_episodes*self.max_steps*self.n_stocks + (self.count-j)*self.n_stocks
            df_new = self.df.iloc[start_point:start_point+10]
            df_new = df_new.iloc[:,[1,4]] 
            #print(self.count, df_new)
            obs = [1]
            for i in range(self.n_stocks):
                #print(line)
                obs.append(df_new.iloc[i, 1]/df_new.iloc[i, 0])
            #print(obs)
            state.append(np.array(obs))
        #print(np.array(state))
        return np.array(state)


        #start_point = self.count_episodes*self.max_steps*self.n_stocks + self.count*self.n_stocks
        #df_new = self.df.iloc[start_point:start_point+10]
        #df_new = df_new.iloc[:,[1,4]] 
        #print(self.count, df_new)
        #obs = [1]
        #for i in range(self.n_stocks):
        #    #print(line)
        #    obs.append(df_new.iloc[i, 1]/df_new.iloc[i, 0])
        #print(obs)
        #state.append(obs)

        #self.holdings = self.holdings - 
        #new_action = normalizeAction(action)

        return []

    def calculate_reward(self, action):
        #self.state = self.observation_space.sample()
        #print(self.state)
        reward = self.beta*np.dot(self.state[-1], action)
        done = False
        if(self.count>=self.max_steps):
            done = True
        #print("REWARD ", reward)
        return reward, done
        #valueOfHolding = data["Close"]
        #self.portifolio = valueOfHolding*self.holdings
        



    def step(self, action):
        action = self.normalizeAction(action)
        self.state = self.receive_state()
        #print(state)
        self.count +=1

        reward, done = self.calculate_reward(action)
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

        return self.state.flatten(), reward, done, []


   
    def readData(self):
      ficheiro = open('gym_anytrading/datasets/data/STOCKS_AMBEV.csv', 'r')
      reader = csv.DictReader(ficheiro, delimiter = ',')
      #print(reader)
      #for linha in reader:
      #    print (linha["Close"])
      return reader