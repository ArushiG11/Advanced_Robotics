import numpy as np
from Actor_Critic import Agent
import matplotlib.pyplot as plt
from ball_navigation_env import BallNavigationEnv

# def plot_learning_curve(x, scores, filename):
    
#     running_avg = np.zeros_like(scores)
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.title('Running Average of Previous 100 Scores')
#     plt.savefig(filename)

# if __name__ == '__main__':
#     env = BallNavigationEnv('nav1.xml')
#     agent = Agent(alpha=0.0003, gamma=0.99, n_actions=env.n_actions)
#     n_games = 1000
#     scores = []

#     for i in range(n_games):
#         observation = env.reset()
#         done = False
#         score = 0
#         while not done:
#             action, log_prob = agent.choose_action(observation)
#             observation_, reward, done = env.step(action)
#             agent.learn(observation, reward, observation_, done, log_prob)
#             observation = observation_
#             score += reward
#         scores.append(score)
#         print(f'Episode {i}, Score: {score}, Avg Score: {np.mean(scores[-100:])}')

#     x = [i+1 for i in range(len(scores))]
#     plot_learning_curve(x, scores, 'learning_curve.png')
#     env.close()




# def plot_learning_curve(x, scores, filename):
#     running_avg = np.zeros_like(scores)
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.title('Running average of previous 100 scores')
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#     plt.savefig(filename)
#     plt.close()

# if __name__ == '__main__':
#     env = BallNavigationEnv('nav1.xml')
#     agent = Agent(alpha=0.0003, gamma=0.99, n_actions=env.n_actions)
#     n_games = 1000
#     max_steps = 200  # Maximum steps per episode
#     scores = []

#     for i in range(n_games):
#         observation = env.reset()
#         done = False
#         score = 0
#         step_count = 0
        
#         while not done and step_count < max_steps:
#             action, log_prob = agent.choose_action(observation)
#             observation_, reward, done = env.step(action)
#             agent.learn(observation, reward, observation_, done, log_prob)
#             observation = observation_
#             score += reward
#             step_count += 1
            
#         scores.append(score)
        
#         if (i + 1) % 10 == 0:
#             print(f'Episode {i+1}, Score: {score:.2f}, Average Score: {np.mean(scores[-100:]):.2f}')

#     x = [i+1 for i in range(len(scores))]
#     plot_learning_curve(x, scores, 'learning_curve.png')
#     env.close()


import tensorflow as tf
import tensorflow_probability  as tfp
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import time

class RobotEnvironment(gym.Env):
    def __init__(self, model_path):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Set goal position (can be modified as needed)
        self.goal_position = np.array([0.9, 0.0])  # Matching the target4 position in MuJoCo model
        self.goal_threshold = 0.1  # Epsilon for goal reaching
        self.dt = 0.1  # Time step
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Random initial position
        init_x = np.random.uniform(-0.1, 1.0)
        init_y = np.random.uniform(-0.3, 0.3)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = init_x
        self.data.qpos[1] = init_y
        self.data.qvel[:] = 0.0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        pos = self.data.qpos[:2]
        vel = self.data.qvel[:2]
        return np.concatenate([pos, vel])
    
    def step(self, action):
        # Apply action forces with noise
        noise_x = np.random.normal(0, 0.1)
        noise_y = np.random.normal(0, 0.1)
        
        self.data.ctrl[0] = action[0] + noise_x
        self.data.ctrl[1] = action[1] + noise_y
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get new state
        obs = self._get_obs()
        
        # Calculate reward
        dist_to_goal = np.linalg.norm(obs[:2] - self.goal_position)
        reward = 1.0 if dist_to_goal <= self.goal_threshold else 0.0
        
        # Check if done
        done = False  # Episode continues until time limit
        
        return obs, reward, done, False, {}

class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_dim * 2)  # Mean and log_std
        ])
        
        # Critic network (value function)
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def get_action(self, state):
        """Get action and its parameters from the current policy."""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_params = self.actor(state_tensor)[0]  # Get first batch item
        
        # Split into means and log_stds
        means, log_stds = tf.split(action_params, 2)
        stds = tf.exp(log_stds)
        
        # Sample from normal distribution
        dist = tfp.distributions.Normal(means, stds)
        actions = dist.sample()
        
        # Clip actions to [-1, 1]
        actions = tf.clip_by_value(actions, -1.0, 1.0)
        
        return actions.numpy(), means.numpy(), stds.numpy()
    
    def get_value(self, state):
        """Get value estimate for the given state."""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.critic(state_tensor)[0, 0]

    def get_action_params(self, state):
        """Get the raw action parameters (means and log_stds)."""
        action_params = self.actor(state)
        means, log_stds = tf.split(action_params, 2, axis=-1)
        return means, log_stds

def train_ac(env, model, episodes=1000, max_steps=200):
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate=0.001)
    gamma = 0.99
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            # Get action and value
            action, action_mean, action_std = model.get_action(state)
            value = model.get_value(state)

            # Take action in environment
            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
            episode_reward += reward

            # Get next state value
            next_value = model.get_value(next_state)

            # Calculate TD error
            td_target = reward + gamma * next_value * (1 - float(done))
            td_error = td_target - value

            # Update networks
            with tf.GradientTape() as tape_actor:
                action_params = model.actor(state_tensor)
                means, log_stds = tf.split(action_params, 2, axis=-1)
                stds = tf.exp(log_stds)
                dist = tfp.distributions.Normal(means, stds)
                action_tensor = tf.convert_to_tensor([action], dtype=tf.float32)
                log_prob = tf.reduce_sum(dist.log_prob(action_tensor), axis=-1)
                actor_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(td_error))

            with tf.GradientTape() as tape_critic:
                value_pred = model.critic(state_tensor)
                critic_loss = tf.reduce_mean(tf.square(td_target - value_pred))

            actor_grads = tape_actor.gradient(actor_loss, model.actor.trainable_variables)
            critic_grads = tape_critic.gradient(critic_loss, model.critic.trainable_variables)

            optimizer_actor.apply_gradients(zip(actor_grads, model.actor.trainable_variables))
            optimizer_critic.apply_gradients(zip(critic_grads, model.critic.trainable_variables))

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}")

    return episode_rewards

# Main execution
def main():
    env = RobotEnvironment("Nav1.xml")
    model = ActorCritic(state_dim=4, action_dim=2)
    
    # Train the model for 1000 episodes
    rewards = train_ac(env, model, episodes=1000)

    # Plot learning curve for all 1000 episodes
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 1001), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve (1000 Episodes)')
    plt.savefig('learning_curve_1000.png')
    plt.close()

if __name__ == "__main__":
    main()