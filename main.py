import numpy as np
from Actor_Critic import ActorCriticAgent
import matplotlib.pyplot as plt
from ball_navigation_env import BallNavigationEnv

def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Step')
    plt.title('Learning Curve')

def main():
    # Initialize MuJoCo environment and agent
    xml_path = 'nav1.xml'  # Path to your XML file
    env = BallNavigationEnv(xml_path)
    agent = ActorCriticAgent()
    filename = 'actor_critics.png'
    figure_file = 'plots/' + filename

    num_episodes = 10
    max_steps = 2000

    rewards = []
    avg_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        for i in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # print(reward, done)
            agent.learn(state, action, reward, next_state, done, log_prob)
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)
        # print(f"Total Reward: {total_reward}")
        print(f"Total Reward: {avg_reward}")

        # if episode % 50 == 0:
        #     print(f"Episode {episode}, Avg Reward: {avg_rewards[-1]:.2f}")


  
     # Plot learning curve
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Step')
    plt.title('Learning Curve')
    plt.show()


if __name__ == "__main__":
    main()