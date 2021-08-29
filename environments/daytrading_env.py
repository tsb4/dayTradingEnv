from copy import deepcopy
from typing import Union
from warnings import warn

import gym
import numpy as np
import pandas as pd


class DayTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_symbols, date_range, window_length=2, commission_rate=0.00023660):
        self.df = self.__preprocess_df(df, stock_symbols, date_range)
        self.symbols = stock_symbols

        self.W = window_length
        self.N = len(self.symbols) + 1
        self.beta = 1.0 - commission_rate

        self.step_count = None
        self.observations = None
        self.observation = None
        self.total_reward = None
        self.portfolio_value = None
        self.portfolio_value_hist = None
        self.action_hist = None

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(self.W, self.N), dtype=np.float32)

        self.reset()

    def step(self, action: Union[list, np.ndarray]):
        if not self.action_space.contains(action):
            raise ValueError("Action {} does not belong to the action space {}".format(action, self.action_space))

        self.step_count += 1
        self.observation = self.__get_observation()
        reward = self.__calc_reward(self.__normalize_action(action))
        done = (self.step_count + 1) >= len(self.observations)
        info = {
            'total_reward': self.total_reward,
            'portfolio_value': self.portfolio_value
        }

        return self.__normalize_observation(self.observation), reward, done, info

    def reset(self):
        self.step_count = 0
        self.observations = [w for w in self.df.rolling(self.W)]
        self.observation = self.__get_observation()
        self.total_reward = 0
        self.portfolio_value = 1
        self.portfolio_value_hist = pd.Series([1.0], index=[self.df.index[0]], dtype=np.float32)
        self.action_hist = pd.DataFrame([[1.0, *np.zeros(self.N - 1)]], index=[self.df.index[0]],
                                        columns=['$', *self.df.columns], dtype=np.float32)

        return self.__normalize_observation(self.observation)

    def render(self, mode='human'):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(self.observations[self.step_count])

    def __get_observation(self):
        window = self.observations[self.step_count]

        return np.insert(np.concatenate((np.ones((self.W - len(window), self.N - 1)), window), axis=0), 0, 1.0, axis=1)

    def __calc_reward(self, action):
        reward = np.log(self.beta * np.dot(self.observation[-1], action))

        self.total_reward += reward
        self.portfolio_value *= np.dot(([1.0] + [self.beta] * (self.N - 1)) * self.observation[-1], action)
        self.portfolio_value_hist.at[self.df.index[self.step_count]] = self.portfolio_value
        self.action_hist.at[self.df.index[self.step_count]] = action

        return reward

    @staticmethod
    def __normalize_action(action):
        action = np.asarray(action)
        action = (action + 1.0) / 2.0
        action_sum = action.sum()

        if np.isclose(action_sum, [0.0, np.nan], equal_nan=True, atol=1e-16).any():
            warn(f"Received {action_sum}'s as action!")
            action = np.array([1.0, *np.zeros(action.shape[0] - 1)])
            action_sum = 1.0

        return action / action_sum

    @staticmethod
    def __normalize_observation(observation):
        return (observation - 1.0) * 100.0

    @staticmethod
    def __preprocess_df(df, stock_symbols, date_range=None):
        df = deepcopy(df)[df['Symbol'].isin(stock_symbols)].pivot(columns='Symbol')
        df = df['Close'] / df['Open']

        if date_range:
            df = df.loc[slice(*date_range)]

        return df[stock_symbols]
