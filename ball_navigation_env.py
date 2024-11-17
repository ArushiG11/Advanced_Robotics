import mujoco
import numpy as np
import glfw

class BallNavigationEnv:
    def __init__(self, xml_path, goal_pos=[0.9, 0.0], epsilon=0.05):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.goal_pos = np.array(goal_pos)
        self.epsilon = epsilon

        # Action bounds (-1, 1) for x and y forces
        self.action_low = -1
        self.action_high = 1
        self.n_actions = 2

        # State dimensions (x, y, vx, vy)
        self.state_dim = 4

    def step(self, action):
        # Apply action: Set control forces
        self.data.ctrl[0] = np.clip(action[0], self.action_low, self.action_high)
        self.data.ctrl[1] = np.clip(action[1], self.action_low, self.action_high)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get state
        position = self.data.qpos[:2]  # [x, y]
        velocity = self.data.qvel[:2]  # [vx, vy]
        state = np.concatenate([position, velocity])

        # Calculate reward and check if done
        reward = 1 if np.linalg.norm(position - self.goal_pos) <= self.epsilon else 0
        done = reward > 0  # Episode ends when target is reached

        return state, reward, done

    def reset(self):
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial position
        init_pos = np.random.uniform(-0.5, 0.5, size=2)
        self.data.qpos[:] = np.append(init_pos)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        # Return initial state
        position = self.data.qpos[:2]
        velocity = self.data.qvel[:2]
        state = np.concatenate([position, velocity])
        return state

    def render(self):
        if self.renderer is None:
            # Initialize GLFW-based renderer
            if not glfw.init():
                raise Exception("GLFW failed to initialize")

            window = glfw.create_window(800, 600, "Ball Navigation", None, None)
            glfw.make_context_current(window)
            self.renderer = (window, mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150))

        window, mjr_context = self.renderer

        if not glfw.window_should_close(window):
            mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, mujoco.MjvScene())
            mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), mujoco.MjvScene(), mjr_context)
            glfw.swap_buffers(window)
            glfw.poll_events()

    def close(self):
        if self.renderer:
            window, _ = self.renderer
            glfw.destroy_window(window)
            glfw.terminate()
