import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorCriticNetwork

class ActorCriticAgent:
    def __init__(self, alpha=0.001, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.action = None
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        

    def select_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, sigma, _ = self.actor_critic(state)

        mu = tf.cast(mu, tf.float32)
        sigma = tf.cast(sigma, tf.float32)
        
        dist = tfp.distributions.Normal(mu, sigma)
        action = tf.clip_by_value(dist.sample(), -1, 1)
        log_prob = dist.log_prob(action)
        self.action = action
        # print(f"State: {state.numpy()}, Mu: {mu.numpy()}, Sigma: {sigma.numpy()}, Action: {action.numpy()}")
        return action.numpy()[0], log_prob

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, action, reward, state_, done, log_prob):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            mu, sigma, value = self.actor_critic(state)
            _, _, value_ = self.actor_critic(state_)
            value = tf.squeeze(value)
            value_ = tf.squeeze(value_)
            
            delta = reward + self.gamma * value_ * (1 - int(done)) - value
            critic_loss = delta**2

            dist = tfp.distributions.Normal(tf.cast(mu, tf.float32), tf.cast(sigma, tf.float32))
            # log_prob = dist.log_prob(self.action)
            
            actor_loss = -log_prob * delta
            entropy_bonus = tf.reduce_mean(dist.entropy()) 
            actor_loss -= 0.01 * entropy_bonus
            total_loss = critic_loss + actor_loss
        

        grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        # for var, grad in zip(self.actor_critic.trainable_variables, grads):
        #     if grad is None or tf.reduce_mean(grad).numpy() == 0:
        #         print(f"Zero gradient for {var.name}")

        self.actor_critic.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
