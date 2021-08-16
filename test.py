import gym
import gym_anytrading

# Using penalty env
env = gym.make('stocks-v0')

dataset = env.readData()
env.reset()
# Run for 1 episode and print reward at the end
nEpisodes = 1
for line in dataset:
  action = env.action_space.sample()
  next_state, reward, done, _ = env.step(action, line)
  print(reward)
