import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import os

#Asset Paths
ASSET_PATH = os.path.join(os.path.dirname(__file__), "../multidrone/assets/")
platform_path = os.path.abspath(os.path.join(ASSET_PATH, "platform.stl")).replace("\\", "/")
drone_path = os.path.abspath(os.path.join(ASSET_PATH, "quadrotor.urdf")).replace("\\", "/")

class DroneEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.max_steps = 500 #Best from what I saw
        self.reward_track = []
        
        # Connect renderer
        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Establish Gravity
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        #Viewport camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=135,
            cameraPitch=-5.6,
            cameraTargetPosition=[0, 0, 0.5]
            )
        
        # Action space: 4 continuous motor thrusts
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: position (x,y,z), orientation (roll,pitch,yaw)
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32) #Normalized


        # Load Mesh and Collisions
        col_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=platform_path,
            meshScale=[1.5, 1.5, 1.5],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.stage = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0]
        )
        self.drone = p.loadURDF(drone_path)
        print(p.getCollisionShapeData(self.stage, -1))

    def reset(self, *, seed=None, options=None):
        
        # Resestablish Gravity and reset sim
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        #Reset the Mesh and Collisions
        col_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=platform_path,
            meshScale=[1.5, 1.5, 1.5],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.stage = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0]
        )

        # Drone Starting position
        start_pos = [0, 0, 0.25]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.drone = p.loadURDF(drone_path, basePosition=start_pos, baseOrientation=start_ori)
        
        #Reset step counter
        self.step_counter = 0 

        #Get starting observations
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
       
       #Track steps per episode
        self.step_counter += 1 
        
        # Apply motor forces
        for i in range(4):
            p.applyExternalForce(self.drone, -1, [0, 0, action[i] * 100], [0, 0, 0], p.LINK_FRAME)

        # Steps the Simulation to Next State
        p.stepSimulation()
        
        # Slows render so that it can be observed
        if self.render:
            time.sleep(1./240.)
        
        # Get X,Y,Z position
        # Get new obserations after simulation step
        obs = self._get_obs()
        target = np.array([0.0, 0.0, 0.5])  # desired position
        pos = obs[:3]  # assuming obs[0:3] = [x, y, z]
        x,y,z = obs[:3] #Unpack for condition flags later
        
        ########### REWARD FUNCTIONS ###########
        # distance = np.linalg.norm(pos - target) #calculated distance from target
        # reward = 1.0 - distance # Max reward is 1 when distance is at 0. 
        # reward = max(reward, 0.0) #Returns at worst a 0 for reward
        # if distance < 0.05: #Additional Reward for being close to the target
        #     reward += 0.5

        reward = np.exp(-np.linalg.norm(pos - target)) #Exponential Decay Reward as drone moves away from target. 
        if np.linalg.norm(pos - target) < 0.05:
            reward += 10
        reward -= 0.1 * np.linalg.norm(pos[:2])
        if z < 0.2: 
            reward -= 0.1 # penalize crash or too low
        if z > 0.5:
            reward -= (z - target[2]) * 2.0  # penalize overshoot
        ########################################

        # Terminatation positions
        terminated_x = (-1.0 > x) or (x > 1.0)
        terminated_y = (-1.0 > y) or (y > 1.0)
        terminated_z = (z > 1.0)
        terminated = terminated_x or terminated_y or terminated_z
        
        # To control how many steps are okay in an episode. 
        truncated = self.step_counter >= self.max_steps
        
        if terminated or truncated: 
            self.reward_track.append(reward)
        
        # print(f'Step: {self.step_counter}')
        #Check for X movement
        if (x < -1): 
            print('-X-X-X-X-X-X-X')
        elif (x > 1.0): 
            print('+X+X+X+X+X+X+X')
        #Check for Y movement
        if (y < -1): 
            print('-Y-Y-Y-Y-Y-Y-Y')
        elif (y > 1.0): 
            print('+Y+Y+Y+Y+Y+Y+Y')
        #Check for Altitude
        if (z < 0.2): 
            print('💥 CRASH!!!')
        elif (z > 1.0): 
            print('🚀 TOO High!!!')
        print(f'☺️{reward}')
        return obs, reward, terminated, truncated, {}

    def _get_obs(self): #Normalized
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        orn_euler = p.getEulerFromQuaternion(orn)
        
        # Normalize position
        x = np.clip(pos[0] / 1.5, -1.0, 1.0)
        y = np.clip(pos[1] / 1.5, -1.0, 1.0)
        z = np.clip((pos[2] - 1.25) / 1.25, -1.0, 1.0)  # center around z=1.25, scale to [-1,1]
        
        # Normalize orientation
        roll = orn_euler[0] / np.pi
        pitch = orn_euler[1] / np.pi
        yaw = orn_euler[2] / np.pi
        
        # Clip to [-1, 1]
        roll = np.clip(roll, -1.0, 1.0)
        pitch = np.clip(pitch, -1.0, 1.0)
        yaw = np.clip(yaw, -1.0, 1.0)

        return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

    def close(self):
        p.disconnect(self.client)

def plot_list(values, title="List Plot", xlabel="Index", ylabel="Value", filename="plot.png"):
    """
    Plots a list of numeric values and saves it as an image.

    Parameters:
    - values (list): A list of numeric values to plot
    - title (str): Plot title
    - xlabel (str): Label for the x-axis
    - ylabel (str): Label for the y-axis
    - filename (str): Filename to save the plot (e.g., "output.png")
    """
    plt.figure(figsize=(8, 4))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)  # Save the figure

if __name__ == "__main__":
    env = DroneEnv(render=True)
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        print(obs, reward)

