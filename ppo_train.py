from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.pybullet_env import DroneEnv
from datetime import datetime
import torch
import time

start = time.time()

#GPU not good for PPO
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

env = DummyVecEnv([lambda: DroneEnv(render=False)])

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=1_000_000)

end = time.time() 
print(f'Time elapsed: {end-start:.2f} seconds') 

model.save(f"ppo_{datetime.now()}")
env.close()
