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
            cameraDistance=2.5,
            cameraYaw=135,
            cameraPitch=-5.6,
            cameraTargetPosition=[0, 0, 0.5]
            )
        
        # Action space: 4 continuous motor thrusts
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: position (x,y,z), orientation (roll,pitch,yaw)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Load plane and drone
        # self.stage = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/stage.urdf")
        col_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="../assets/platform.stl",
            meshScale=[1.5, 1.5, 1.5],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.stage = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0]
        )
        self.drone = p.loadURDF("../assets/quadrotor.urdf")
        print(p.getCollisionShapeData(self.stage, -1))

    def reset(self, *, seed=None, options=None):
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        #Reset up the stage. 
        # self.stage = p.loadURDF("/Users/chrisguarino/Documents/Programming/multidrone/assets/stage.urdf", useMaximalCoordinates=True)
        col_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="../assets/platform.stl",
            meshScale=[1.5, 1.5, 1.5],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.stage = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0]
        )

        #Reset Up drone
        start_pos = [0, 0, 0.25]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.drone = p.loadURDF("../assets/quadrotor.urdf", basePosition=start_pos, baseOrientation=start_ori)
        
        #Reset step counter
        self.step_counter = 0 

        #Get new observations
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
       
       #Track steps per episode
        self.step_counter += 1 
        
        # Apply motor forces
        for i in range(4):
            p.applyExternalForce(self.drone, -1, [0, 0, action[i] * 100], [0, 0, 0], p.LINK_FRAME)

        # This enacts the next frame in the simulation
        p.stepSimulation()
        # Slows render so that it can be observed
        if self.render:
            time.sleep(1./240.)
        
        # Get X,Y,Z position
        # Get new obserations after simulation step
        obs = self._get_obs()
        target = np.array([0.0, 0.0, 1.0])  # desired position
        pos = obs[:3]  # assuming obs[0:3] = [x, y, z]
        reward = -np.linalg.norm(pos - target)

        # Terminatation positions
        x,y,z = obs[:3]
        terminated_x = (-1.0 > x) or (x > 1.0)
        terminated_y = (-1.0 > y) or (y > 1.0)
        terminated_z = (z < 0.1) or (z > 2.0)
        terminated = terminated_x or terminated_y or terminated_z
        
        print(f'Step: {self.step_counter}')
        
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
        if (z < 0.1): 
            print('💥 CRASH!!!')
        elif (z > 2.0): 
            print('🚀 TOO High!!!')
        
        # To control how many steps are okay in an episode. 
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
