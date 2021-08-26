import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        self.__deque = deque([], size)

    def append(self, observation, action, reward, next_observation, done):
        self.__deque.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size):
        batch = random.sample(self.__deque, min(batch_size, len(self.__deque)))

        observations = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_observations = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.__deque)

    def clear(self):
        self.__deque.clear()
