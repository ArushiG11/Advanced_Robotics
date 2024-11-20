import numpy as np
from Actor_Critic import ActorCriticAgent
import matplotlib.pyplot as plt
from ball_navigation_env import BallNavigationEnv

def main():
    # Initialize MuJoCo environment and agent
    xml_path = 'nav1.xml'  # Path to your XML file
    env = BallNavigationEnv(xml_path)
    agent = ActorCriticAgent()

    num_episodes = 200
    max_steps = 200
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if not load_checkpoint:
                agent.learn(state, action, reward, next_state, done, log_prob)
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward / max_steps)
        if episode % 50 == 0:
            print(f"Episode {episode}, Avg Reward: {rewards[-1]:.2f}")

    # Plot learning curve
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Step')
    plt.title('Learning Curve')
    plt.show()


if __name__ == "__main__":
    main()
