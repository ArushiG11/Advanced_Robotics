import numpy as np
from Actor_Critic import Agent
import matplotlib.pyplot as plt
from ball_navigation_env import BallNavigationEnv

def plot_learning_curve(x, scores, filename):
    
    running_avg = np.zeros_like(scores)
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running Average of Previous 100 Scores')
    plt.savefig(filename)

if __name__ == '__main__':
    env = BallNavigationEnv('nav1.xml')
    agent = Agent(alpha=0.0003, gamma=0.99, n_actions=env.n_actions)
    n_games = 1000
    scores = []

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, log_prob = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            agent.learn(observation, reward, observation_, done, log_prob)
            observation = observation_
            score += reward
        scores.append(score)
        print(f'Episode {i}, Score: {score}, Avg Score: {np.mean(scores[-100:])}')

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, 'learning_curve.png')
    env.close()
