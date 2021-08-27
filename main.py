from os import makedirs, path, listdir

import gym
import matplotlib.pyplot as plt

from environments.dataset import B3_STOCKS, top_ten_stocks
from model.ddpg_agent import DDPGAgent


def __main__():
    year = 2017
    episodes = 100

    index, last_ep = ddpg_train(year, episodes)
    ddpg_test(year, index, last_ep)


def ddpg_train(year: int, episodes: int = 100):
    env = gym.make('DayTrading-v0', df=B3_STOCKS, stock_symbols=top_ten_stocks[str(year)],
                   date_range=(f"{year - 2}-01-01", f"{year - 1}-12-31"), window_length=3)

    last_ep = 0
    reward_history = []
    test_portfolio_values = []
    save_dir, index = __get_save_dir()
    ddpg_agent = DDPGAgent(env)
    max_step = len(env.df) - 1

    try:
        for ep in range(1, episodes + 1):
            observation = env.reset()
            done = False
            info = None

            title = f"Training {index}: Episode {ep}"
            subtitle = ''
            progress = ''

            while not done:
                action = ddpg_agent.predict(observation, add_noise=True)
                next_observation, reward, done, info = env.step(action)

                ddpg_agent.remember(observation, action, reward, next_observation, done)
                ddpg_agent.learn()

                observation = next_observation

                subtitle = f"Portfolio value: {info['portfolio_value']:.5f}"
                progress = f"{title} {env.step_count}/{max_step} - {subtitle}"
                print(f"\r{progress}", end='', flush=True)
            else:
                print('\r', ' ' * len(progress), end='', flush=True)

            print(f"\r{title} - {subtitle}")

            ddpg_agent.save_weights(path.join(save_dir, 'weights', f"ep{ep}"))
            reward_history.append(info['total_reward'])

            __plot(f"Training {index}: Reward history", '', reward_history, path.join(save_dir, 'reward_history.png'))
            __plot(title, subtitle, env.portfolio_value_hist, path.join(save_dir, 'train_plots', f"ep{ep}.png"))

            test_portfolio_values.append(ddpg_test(year, index, ep, ddpg_agent, training=True))

            __plot(f"Testing {index}: Portfolio value history", '', test_portfolio_values,
                   path.join(save_dir, 'portfolio_value_history.training.png'))

            last_ep = ep
    except KeyboardInterrupt:
        print(f'\nEarly stop before the end of episode {last_ep + 1}')
        index = None
    finally:
        if last_ep > 0:
            open(path.join(save_dir, f"done_in_{last_ep}_episodes"), 'a').close()

    return index, last_ep


def ddpg_test(year: int, index, ep, ddpg_agent: DDPGAgent = None, training=False):
    env = gym.make('DayTrading-v0', df=B3_STOCKS, stock_symbols=top_ten_stocks[str(year)],
                   date_range=(f"{year}-01-01", f"{year}-12-31"), window_length=3)

    if index is None:
        return

    save_dir, index = __get_save_dir(index=index)

    if ddpg_agent is None:
        ddpg_agent = DDPGAgent(env)
        ddpg_agent.load_weights(path.join(save_dir, 'weights', f"ep{ep}"))

    max_step = len(env.df) - 1
    obs = env.reset()
    done = False
    info = None
    subtitle = ''
    progress = ''

    if not training:
        title = f"Testing {index}"
        plot_filename = path.join(save_dir, 'portfolio_value_history.png')
    else:
        title = f"Testing {index}: Episode {ep}"
        plot_filename = path.join(save_dir, 'test_plots', f"ep{ep}.png")

    while not done:
        action = ddpg_agent.predict(obs)
        obs, reward, done, info = env.step(action)

        subtitle = f"Portfolio value: {info['portfolio_value']:.5f}"
        progress = f"{title} {env.step_count}/{max_step} - {subtitle}"
        print(f"\r{progress}", end='', flush=True)
    else:
        print('\r', ' ' * len(progress), end='', flush=True)

    print(f"\r{title} - {subtitle}")

    __plot(title, subtitle, env.portfolio_value_hist, plot_filename, color='r')

    return info['portfolio_value']


def __get_save_dir(out_dir='out.ddpg', index=None):
    makedirs(out_dir, exist_ok=True)

    if index is None:
        dirs = [e for e in listdir(out_dir) if path.isdir(path.join(out_dir, e)) and e.isdigit()]
        dirs = sorted([int(d) for d in dirs if any('done' in e for e in listdir(path.join(out_dir, d)))])

        index = '1' if len(dirs) == 0 else str(dirs[-1] + 1)

    save_dir = path.join(out_dir, str(index))
    makedirs(save_dir, exist_ok=True)

    return save_dir, index


def __plot(title, subtitle, data, filename, color=None, show=True):
    makedirs(path.dirname(filename), exist_ok=True)

    plt.suptitle(title, fontsize=12)
    plt.title(subtitle, fontsize=9)
    plt.xticks(rotation=25)
    plt.grid()
    plt.plot(data, color=color)
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()


if __name__ == '__main__':
    __main__()
