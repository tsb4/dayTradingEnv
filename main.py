from os import makedirs, path, listdir

import gym
import matplotlib.pyplot as plt

from environments.dataset import B3_STOCKS, top_ten_stocks
from model.ddpg_agent import DDPGAgent


def __main__():
    year = 2017
    episodes = 100

    index, ep = ddpg_train(year, episodes)
    ddpg_test(year, index, ep)


def ddpg_train(year: int, episodes: int = 100):
    env = gym.make('DayTrading-v0', df=B3_STOCKS, stock_symbols=top_ten_stocks[str(year)],
                   date_range=(f"{year - 2}-01-01", f"{year - 1}-12-31"), window_length=3)

    ep = 0
    reward_history = []
    save_dir, index = __get_save_dir()
    ddpg_agent = DDPGAgent(env)

    try:
        for ep in range(1, episodes + 1):
            observation = env.reset()
            done = False
            info = None

            while not done:
                action = ddpg_agent.predict(observation, add_noise=True)
                next_observation, reward, done, info = env.step(action)

                ddpg_agent.remember(observation, action, reward, next_observation, done)
                ddpg_agent.learn()

                observation = next_observation
                print('.' if env.step_count % 6 == 0 else '', end='', flush=True)

            reward_history.append(info['total_reward'])

            title = f"Training {index}: Episode {ep}"
            subtitle = f"Portfolio value: {info['portfolio_value']}"

            print(f"\n{title} - {subtitle}")

            __plot(f"Training {index}: Reward History", '', reward_history, path.join(save_dir, 'reward_hist.png'))
            __plot(title, subtitle, env.portfolio_value_hist, path.join(save_dir, 'train_plots', f"ep{ep}.png"))

            ddpg_agent.save_weights(path.join(save_dir, 'weights', f"ep{ep}"))

            if ep == 1 or ep % 5 == 0:
                ddpg_test(year, index, ep, ddpg_agent)
    except KeyboardInterrupt:
        print(f'\nEarly stop in episode {ep}')
        ep -= 1

    if ep > 0:
        open(path.join(save_dir, f"done_training_in_{ep}_episodes"), 'a').close()
    else:
        index = None

    return index, ep


def ddpg_test(year: int, index, ep, ddpg_agent: DDPGAgent = None):
    env = gym.make('DayTrading-v0', df=B3_STOCKS, stock_symbols=top_ten_stocks[str(year)],
                   date_range=(f"{year}-01-01", f"{year}-12-31"), window_length=3)

    if index is None:
        return

    save_dir, index = __get_save_dir(index=index)
    train_test = True

    if ddpg_agent is None:
        train_test = False

        ddpg_agent = DDPGAgent(env)
        ddpg_agent.load_weights(path.join(save_dir, 'weights', f"ep{ep}"))

    obs = env.reset()
    done = False
    info = None

    while not done:
        action = ddpg_agent.predict(obs)
        obs, reward, done, info = env.step(action)

    if not train_test:
        title = f"Testing {index}"
        subtitle = f"Portfolio value: {info['portfolio_value']}"
        plot_filename = path.join(save_dir, 'portfolio_value_hist.png')
    else:
        title = f"Testing {index}: Episode {ep}"
        subtitle = f"Portfolio value: {info['portfolio_value']}"
        plot_filename = path.join(save_dir, 'test_plots', f"ep{ep}.png")

    print(f"{title} - {subtitle}")
    __plot(title, subtitle, env.portfolio_value_hist, plot_filename)


def __get_save_dir(out_dir='out.ddpg', index=None):
    makedirs(out_dir, exist_ok=True)

    if index is None:
        dirs = sorted([e for e in listdir(out_dir) if path.isdir(path.join(out_dir, e))])
        dirs = sorted([d for d in dirs if any('done' in e for e in listdir(path.join(out_dir, d)))])

        index = '1' if len(dirs) == 0 else str(int(dirs[-1]) + 1)

    save_dir = path.join(out_dir, str(index))
    makedirs(save_dir, exist_ok=True)

    return save_dir, index


def __plot(title, subtitle, data, filename):
    makedirs(path.dirname(filename), exist_ok=True)

    plt.suptitle(title, fontsize=12)
    plt.title(subtitle, fontsize=9)
    plt.xticks(rotation=25)
    plt.grid()
    plt.plot(data)
    plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == '__main__':
    __main__()
