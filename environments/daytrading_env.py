from copy import deepcopy
from typing import Union

import gym
import numpy as np
import pandas as pd


class DayTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_symbols, date_range, W=2, commission_rate=0.00023660):
        self.df = self.__preprocess_df(df, stock_symbols, date_range)
        self.symbols = stock_symbols

        self.W = W
        self.N = len(self.symbols) + 1
        self.beta = 1 - commission_rate

        self.__windows = []
        self.step_count = 0
        self.state = None

        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(self.N,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=(self.W, self.N), dtype=np.float32)

        self.reset()

    def step(self, action: Union[list, np.ndarray]):
        if not self.action_space.contains(action):
            raise ValueError("Action {} does not belong to the action space {}".format(action, self.action_space))

        action = np.asarray(action)
        action /= sum(action)

        self.step_count += 1
        self.state = self.__get_observation()

        reward = self.__calc_reward(action)
        done = (self.step_count + 1) >= len(self.__windows)

        return self.state, reward, done, {}

    def reset(self):
        self.__windows = [w for w in self.df.rolling(self.W)]
        self.step_count = 0
        self.state = self.__get_observation()

        return self.state

    def render(self, mode='human'):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(self.__windows[self.step_count])

    def __get_observation(self):
        window = self.__windows[self.step_count]

        return np.insert(
            np.concatenate((np.ones((self.W - len(window), self.N - 1)), np.asarray(window)), axis=0),
            0, 1.0, axis=1
        )

    def __calc_reward(self, action):
        return np.log(self.beta * np.dot(self.state[-1], action))

    @staticmethod
    def __preprocess_df(df, stock_symbols, date_range=None):
        df = deepcopy(df)[df['Symbol'].isin(stock_symbols)].pivot(columns='Symbol')
        df = df['Close'] / df['Open']

        if date_range:
            df = df.loc[slice(*date_range)]

        return df[stock_symbols]
