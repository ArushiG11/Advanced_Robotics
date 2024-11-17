import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class ActorCriticNetwork(keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=128, n_actions=2):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.mu = Dense(n_actions, activation='tanh')  # Mean of Gaussian
        # self.sigma = Dense(n_actions, activation='softplus')  # Standard deviation
        self.log_sigma = Dense(n_actions)
        self.v = Dense(1)  # State value

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        log_sigma = tf.clip_by_value(self.log_sigma(x), -2, 2)
        sigma = tf.exp(log_sigma)
        v = self.v(x)
        return mu, sigma, v