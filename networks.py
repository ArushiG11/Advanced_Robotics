import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import numpy as np

class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=128, n_actions=2, name='actor-critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')
        self.v = Dense(1, activation=None)  # Critic output: State value
        self.mu = Dense(n_actions, activation='tanh')  # Actor output: Mean actions
        self.sigma = Dense(n_actions, activation='softplus')  # Ensures sigma is positive

  # Trainable log std deviation

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        value = self.v(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma, value


