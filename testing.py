# testing.py
from stable_baselines3 import PPO
from envs.pybullet_env import DroneEnv  # adjust import to your actual environment file
import time

def main():
    # Create your environment
    env = DroneEnv(render=True)

    # Load trained model
    model = PPO.load("ppo_2025-07-22 21:31:09.289375", env=env)

    # Reset environment
    obs, info = env.reset()
    done = False

    while not done:
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Cleanup
    env.close()

if __name__ == "__main__":
    main()