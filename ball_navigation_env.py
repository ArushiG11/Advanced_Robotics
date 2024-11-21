import gymnasium as gym
import mujoco
import numpy as np

class BallNavigationEnv(gym.Env):
    def __init__(self, xml_path='nav1.xml', goal_coords=np.array([0.9, 0.0]), goal_threshold=0.5):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.goal_coords = goal_coords
        self.goal_threshold = goal_threshold

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        # Randomize the ball's initial position and velocity
        self.data.qpos[:2] = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self.data.qvel[:2] = np.zeros(2, dtype=np.float32)
        mujoco.mj_step(self.model, self.data)
        return self._get_state()

    def step(self, action):
        noise = np.random.normal(0, 0.05, size=2)
        self.data.ctrl[:2] = action + noise

        mujoco.mj_step(self.model, self.data)

        state = self._get_state()
        reward = self._compute_reward(state)
        # print(state, reward)
        done = self._check_termination(state)
        # print(f"State: {state[:2]}, Goal: {self.goal_coords}, Reward: {reward}")
        return state, reward, done, {}

    def _get_state(self):
        # Combine position and velocity into a single state vector
        
        pos = self.data.qpos[:2].astype(np.float32)
        vel = self.data.qvel[:2].astype(np.float32)
        return np.concatenate([pos, vel])

    def _compute_reward(self, state):
        dist = np.linalg.norm(state[:2] - self.goal_coords)
        
        # Reward for reaching the goal
        if dist <= self.goal_threshold:
            return 1.0
        
        # Penalty proportional to distance from the goal
        return 0.0

    def _check_termination(self, state):
        distance = np.linalg.norm(state[:2] - self.goal_coords)
        return distance <= self.goal_threshold
