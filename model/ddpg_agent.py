from os import makedirs

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from model.actor_network import ActorNetwork
from model.critic_network import CriticNetwork
from model.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from model.replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, env: gym.Env, gamma=.99, tau=.001, actor_learning_rate=.0001, critic_learning_rate=.001,
                 buffer_size=1000000, batch_size=64, lstm_units=50, fc1_units=50, fc2_units=25, ou_sigma=.3,
                 weights_dir='out.ddpg'):
        self.env = env
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(env.action_space.shape), sigma=ou_sigma)

        self.actor = ActorNetwork(env.action_space.shape[0], lstm_units=lstm_units,
                                  fc1_units=fc1_units, fc2_units=fc2_units, weights_dir=weights_dir)
        self.critic = CriticNetwork(lstm_units=lstm_units, fc1_units=fc1_units, fc2_units=fc2_units,
                                    weights_dir=weights_dir)

        self.target_actor = ActorNetwork(**self.actor.get_config(), name='target_actor')
        self.target_critic = CriticNetwork(**self.critic.get_config(), name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=actor_learning_rate))
        self.critic.compile(optimizer=Adam(learning_rate=critic_learning_rate))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_learning_rate))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_learning_rate))

        self.update_target_networks(tau=1)

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau

        self.target_actor.set_weights(
            [(tau * weight + (1 - tau) * target_weight)
             for weight, target_weight in zip(self.actor.weights, self.target_actor.weights)]
        )

        self.target_critic.set_weights(
            [(tau * weight + (1 - tau) * target_weight)
             for weight, target_weight in zip(self.critic.weights, self.target_critic.weights)]
        )

    def remember(self, observation, action, reward, next_observation, done):
        self.replay_buffer.append(observation, action, reward, next_observation, done)

    def predict(self, observation, add_noise=False):
        action = tf.gather(self.actor(np.expand_dims(observation, axis=0)), 0)

        if add_noise:
            action = tf.clip_by_value(action + self.action_noise(), -1, 1)

        return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_observations)
            next_critic_values = tf.squeeze(self.target_critic((next_observations, target_actions)), axis=1)
            critic_values = tf.squeeze(self.critic((observations, actions)), axis=1)
            target = rewards + self.gamma * next_critic_values * (1 - dones)
            critic_loss = keras.losses.MSE(target, critic_values)

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(observations)
            actor_loss = -self.critic((observations, new_policy_actions))
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_target_networks()

    def save_weights(self, weights_dir=None):
        if weights_dir is not None:
            makedirs(weights_dir, exist_ok=True)

        self.actor.save_weights(weights_dir)
        self.critic.save_weights(weights_dir)
        self.target_actor.save_weights(weights_dir)
        self.target_critic.save_weights(weights_dir)

    def load_weights(self, weights_dir=None):
        observation_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape

        self.actor(np.zeros((1, *observation_shape)))
        self.critic((np.zeros((1, *observation_shape)), np.zeros((1, *action_shape))))
        self.target_actor(np.zeros((1, *observation_shape)))
        self.target_critic((np.zeros((1, *observation_shape)), np.zeros((1, *action_shape))))

        self.actor.load_weights(weights_dir)
        self.critic.load_weights(weights_dir)
        self.target_actor.load_weights(weights_dir)
        self.target_critic.load_weights(weights_dir)
