from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.pybullet_env import DroneEnv
from datetime import datetime

env = DummyVecEnv([lambda: DroneEnv(render=False)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

model.save(f"ppo_{datetime.now()}")
env.close()
