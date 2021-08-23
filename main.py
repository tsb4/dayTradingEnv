import gym
import numpy as np

from environments.dataset import B3_STOCKS, top_ten_stocks


def __main__():
    np.set_printoptions(linewidth=np.inf)
    env = gym.make('DayTrading-v0', df=B3_STOCKS, stock_symbols=top_ten_stocks['2017'],
                   date_range=('2015-01-01', '2015-01-06'), W=3)

    print('---------- Initial state ----------')
    state = env.reset()
    env.render()
    print(state, '\n')

    print('\n---------- Next state ----------')
    state, reward, done, _ = env.step(np.asarray([1.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0]))
    print(f"Reward: {reward}, Done? {done}\n")
    env.render()
    print(state, '\n')

    print('\n---------- Next state ----------')
    state, reward, done, _ = env.step(np.asarray([.0, 1.0, .0, .0, .0, .0, .0, .0, .0, .0, .0]))
    print(f"Reward: {reward}, Done? {done}\n")
    env.render()
    print(state, '\n')


if __name__ == '__main__':
    __main__()
