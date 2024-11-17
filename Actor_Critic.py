import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfp
from networks import ActorCriticNetwork

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu, sigma, _ = self.actor_critic(state)
        dist = tfp.Normal(mu, sigma)
        action = dist.sample()
        action = tf.clip_by_value(action, -1, 1)  # Ensure actions stay within bounds
        log_prob = dist.log_prob(action)
        return action[0].numpy(), log_prob


    def learn(self, state, reward, state_, done, log_prob):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            mu, sigma, value = self.actor_critic(state)
            _, _, next_value = self.actor_critic(state_)

            value = tf.squeeze(value)
            next_value = tf.squeeze(next_value)

            delta = reward + self.gamma * next_value * (1 - int(done)) - value
            critic_loss = delta**2

            dist = tfp.Normal(mu, sigma)
            actor_loss = -log_prob * delta
            total_loss = actor_loss + critic_loss

        gradients = tape.gradient(total_loss, self.actor_critic.trainable_variables)

        # Debug missing gradients
        for var, grad in zip(self.actor_critic.trainable_variables, gradients):
            if grad is None:
                print(f"Gradient missing for variable: {var.name}")

        self.actor_critic.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))
