import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class DroneEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.max_steps = 200
        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        #Viewport camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=25,
            cameraYaw=90,
            cameraPitch=-70,
            cameraTargetPosition=[0, 0, 0.5]
            )
        # Action space: 4 continuous motor thrusts
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: position (x,y,z), orientation (roll,pitch,yaw)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Load plane and drone
        self.plane = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/maze.urdf")
        self.drone = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/quadrotor.urdf", [0, 0, 1])

    def reset(self, *, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        start_pos = [8.5, 1.5, -6.5]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.plane = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/maze.urdf")
        self.drone = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/quadrotor.urdf", basePosition=start_pos, baseOrientation=start_ori)
        self.step_counter = 0 #To truncate an episode

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_counter += 1 #Start counting how long the episode is
        # Apply motor forces
        for i in range(4):
            p.applyExternalForce(self.drone, -1, [0, 0, action[i] * 10], [0, 0, 0], p.LINK_FRAME)

        p.stepSimulation()
        if self.render:
            time.sleep(1./240.)

        obs = self._get_obs()

        z = obs[2]  # current altitude
        reward = -abs(z - 1.0)  # reward for staying close to z=1.0

        # Terminate if drone crashes or flies too high
        terminated = (z < 0.1) or (z > 2.0)
        truncated = self.step_counter >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        orn_euler = p.getEulerFromQuaternion(orn)
        return np.array(list(pos) + list(orn_euler[:3]), dtype=np.float32)

    def close(self):
        p.disconnect(self.client)

if __name__ == "__main__":
    env = DroneEnv(render=True)
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        print(obs, reward)
