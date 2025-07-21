import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.pybullet_env import DroneEnv  # 👈 your DroneEnv code, save as drone_env.py

# Create environment
env = DroneEnv(render=True)

# ✅ Check that your environment is Gym-compatible
check_env(env)

# Instantiate the PPO agent
model = PPO(
    "MlpPolicy",        # Use a simple MLP policy
    env,
    verbose=1,
    tensorboard_log="./ppo_drone_tensorboard/"  # Optional: for TensorBoard
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained agent
model.save("ppo_drone_policy")

env.close()
