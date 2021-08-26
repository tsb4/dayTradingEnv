from os import path

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense, LSTM


class CriticNetwork(keras.Model):
    def __init__(self, lstm_units=100, fc1_units=100, fc2_units=50, name='critic', weights_dir='out.ddpg'):
        super(CriticNetwork, self).__init__()
        self.__name = name
        self.__weights_dir = weights_dir
        self.__config = {'lstm_units': lstm_units, 'fc1_units': fc1_units, 'fc2_units': fc2_units,
                         'weights_dir': weights_dir}

        self.flatten = Flatten()
        self.fc1_1 = Dense(fc1_units, activation='relu')
        self.fc1_2 = Dense(fc1_units, activation='relu')
        self.lstm = LSTM(lstm_units)
        self.fc2 = Dense(fc2_units, activation='relu')
        self.q = Dense(1)

    def call(self, inputs, training=None, mask=None):
        observations, actions = inputs  # (?, 3, 11), (?, 11)

        x = self.flatten(observations)  # (?, 3, 11) -> (?, 33)
        x = tf.concat([x, actions], 1)  # (?, 33), (?, 11) -> (?, 44)
        x = self.fc1_1(x)  # (?, 44) -> (?, 100)

        y = self.fc1_2(observations)  # (?, 3, 11) -> (?, 3, 100)
        y = self.lstm(y)  # (?, 3, 100) -> (?, 100)

        x = tf.concat([x, y], 1)  # (?, 100), (?, 100) -> (?, 200)
        x = self.fc2(x)  # (?, 200) -> (?, 50)
        x = self.q(x)  # (?, 50) -> (?, 1)

        return x

    def save_weights(self, out_dir=None, **kwargs):
        super().save_weights(path.join(out_dir or self.__weights_dir, self.__name + '.h5'), **kwargs)

    def load_weights(self, in_dir=None, **kwargs):
        super().load_weights(path.join(in_dir or self.__weights_dir, self.__name + '.h5'), **kwargs)

    def get_config(self):
        return self.__config
