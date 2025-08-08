import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco_viewer
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class DroneEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.max_steps = 500
        self.reward_track = []
        self.render_mode = render

        # Paths
        base_dir = Path(__file__).parent
        self.model_path = str(base_dir / "assets" / "quadrotor.xml")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Action space: 4 continuous thrust commands from agent
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: normalized (x, y, z, roll, pitch, yaw)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Viewer will be created only if render=True
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1

        # --- Convert agent's 4 thrust actions into rotor forces ---
        max_thrust = 5.0  # N
        thrusts = np.clip((action + 1) / 2 * max_thrust, 0, max_thrust)
        
        # Add torque control — just like before
        torque_dirs = np.array([+1, -1, +1, -1], dtype=np.float32)
        torque_coeff = 0.02  # Tune this if too wobbly
        torques = torque_dirs * thrusts * torque_coeff
        
        # Create full control array (8 actuators: 4 thrust, 4 torque)
        ctrl = np.zeros(8, dtype=np.float32)
        ctrl[:4] = thrusts  # first 4: thrust motors, last 4)
        ctrl[4:8] = torques # last 4: torque motors
        self.data.ctrl[:] = ctrl

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Observation
        obs = self._get_obs()
        target = np.array([0.0, 0.0, 0.5])
        pos = obs[:3]
        x, y, z = pos

        # Reward shaping
        dist_to_target = np.linalg.norm(pos - target)
        reward = np.exp(-dist_to_target)
        reward -= 0.1 * np.linalg.norm(pos[:2])

        if z < 0.2:
            reward -= 0.1
        if z > 0.5:
            reward -= (z - target[2]) * 2.0

        reward -= 0.001 * self.step_counter

        if dist_to_target < 0.05:
            reward += max(2.0 - 0.01 * self.step_counter, 0)
        if dist_to_target < 0.02:
            reward += 0.2
            vel = self.data.qvel[:3]
            if np.linalg.norm(vel) < 0.05:
                reward += 0.2

        terminated = (
            (x < -1.0) or (x > 1.0) or
            (y < -1.0) or (y > 1.0) or
            (z < -1.0) or (z > 1.0)
        )
        truncated = self.step_counter >= self.max_steps

        if terminated or truncated:
            self.reward_track.append(reward)

        print(f'🟢\nX: {obs[0]}, Y: {obs[1]}, Z: {obs[2]}, Roll: {obs[3]}, Pitch: {obs[4]}, Yaw: {obs[5]}, Reward: {reward}\n🔴')
        return obs, reward, terminated, truncated, {}



    def _get_obs(self):
        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        orn_euler = R.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_euler('xyz', degrees=False)

        x = np.clip(pos[0] / 1.5, -1.0, 1.0)
        y = np.clip(pos[1] / 1.5, -1.0, 1.0)
        z = np.clip((pos[2] - 1.25) / 1.25, -1.0, 1.0)

        roll = np.clip(orn_euler[0] / np.pi, -1.0, 1.0)
        pitch = np.clip(orn_euler[1] / np.pi, -1.0, 1.0)
        yaw = np.clip(orn_euler[2] / np.pi, -1.0, 1.0)

        return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

    def render(self):
        if self.render_mode and self.viewer is not None:
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

if __name__ == "__main__":
    env = DroneEnv(render=False)
    obs, _ = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        env.render()
        if term or trunc:
            obs, _ = env.reset()

