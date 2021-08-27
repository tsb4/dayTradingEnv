from os import path

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM


class ActorNetwork(keras.Model):
    def __init__(self, out_units, lstm_units=100, fc1_units=100, fc2_units=50, name='actor', weights_dir='out.ddpg'):
        super(ActorNetwork, self).__init__()
        self.__name = name
        self.__weights_dir = weights_dir
        self.__config = {'out_units': out_units, 'lstm_units': lstm_units,
                         'fc1_units': fc1_units, 'fc2_units': fc2_units, 'weights_dir': weights_dir}

        self.fc1 = Dense(fc1_units, activation='relu')
        self.lstm = LSTM(lstm_units)
        self.fc2 = Dense(fc2_units, activation='relu')
        self.mu = Dense(out_units, activation='tanh')

    def call(self, observations, training=None, mask=None):
        x = self.fc1(observations)  # (?, 3, 11) -> (?, 3, 100)
        x = self.lstm(x)  # (?, 3, 100) -> (?, 100)
        x = self.fc2(x)  # (?, 100) -> (?, 50)
        x = self.mu(x)  # (?, 50) -> (?, 11)

        return x

    def save_weights(self, out_dir=None, **kwargs):
        super().save_weights(path.join(out_dir or self.__weights_dir, self.__name + '.h5'), **kwargs)

    def load_weights(self, in_dir=None, **kwargs):
        super().load_weights(path.join(in_dir or self.__weights_dir, self.__name + '.h5'), **kwargs)

    def get_config(self):
        return self.__config
