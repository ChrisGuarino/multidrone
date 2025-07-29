import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class MultiDroneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, num_drones=2, render_mode=None):
        super(MultiDroneEnv, self).__init__()
        self.num_drones = num_drones
        self.action_space = spaces.MultiDiscrete([5] * num_drones)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(num_drones, 2), dtype=np.float32
        )

        self.state = None
        self.render_mode = render_mode
        self.viewer_initialized = False

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.random.rand(self.num_drones, 2) * 10
        return self.state, {}

    def step(self, action):
        for i, a in enumerate(action):
            if a == 1:   # up
                self.state[i, 1] += 1
            elif a == 2: # down
                self.state[i, 1] -= 1
            elif a == 3: # left
                self.state[i, 0] -= 1
            elif a == 4: # right
                self.state[i, 0] += 1
        self.state = np.clip(self.state, 0, 10)
        reward = -np.sum(np.abs(self.state - 5))
        done = False
        return self.state, reward, done, False, {}

    def render(self):
        if not self.viewer_initialized:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.viewer_initialized = True

        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_title("Multi-Drone Environment")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        # plot drones
        for pos in self.state:
            self.ax.plot(pos[0], pos[1], 'bo', markersize=10)

        plt.pause(0.01)

    def close(self):
        if self.viewer_initialized:
            plt.ioff()
            plt.close()
            self.viewer_initialized = False
