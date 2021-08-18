import gym
import gym_anytrading

# Using penalty env
env = gym.make('trading-v0')

#dataset = env.readData()
# Run for 1 episode and print reward at the end
nEpisodes = 98
for i in range(nEpisodes):
  env.reset()
  done = False
  while not done:
    action = env.action_space.sample()
    #print(action)
    next_state, reward, done, _ = env.step(action)
    print(reward)
