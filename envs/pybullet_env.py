import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class DroneEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # Action space: 4 motor thrusts
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: position (x,y,z), orientation (roll,pitch,yaw), linear and angular velocities
        obs_dim = 12
        high = np.ones(obs_dim) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Load plane
        self.plane = p.loadURDF("plane.urdf")
        # Load your drone URDF here:
        self.drone = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/quadrotor.urdf", [0, 0, 1])

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone, [0,0,1], [0,0,0,1])
        p.resetBaseVelocity(self.drone, [0,0,0], [0,0,0])
        obs = self._get_obs()
        return obs

    def step(self, action):
        # Apply motor forces (you’ll replace this with more realistic thrust control)
        for i in range(4):
            p.applyExternalForce(self.drone, -1, [0,0,action[i]*10], [0,0,0], p.LINK_FRAME)

        p.stepSimulation()
        if self.render:
            time.sleep(1./240.)

        obs = self._get_obs()
        reward = -np.linalg.norm(obs[:3] - np.array([0,0,1]))  # keep near hover point
        done = False
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone)
        orn_euler = p.getEulerFromQuaternion(orn)
        return np.array(list(pos) + list(orn_euler) + list(lin_vel) + list(ang_vel), dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)

if __name__ == "__main__": 
    
    env = DroneEnv(render=True)
    obs = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
