import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env import DroneEnv  # Your environment file
import time

# Function that creates a fresh environment
def make_env(rank):
    def _init():
        return DroneEnv(render=False)  # Rendering off during training
    return _init

if __name__ == "__main__":
    num_envs = 8  # Number of parallel environments
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,     # PPO default, per env
        batch_size=64,
        ent_coef=0.0,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10
    )

    model.learn(total_timesteps=200_000)  # Train for 1M steps total
    model_name = input("Enter model name: ")
    # plot_list(env.reward_track, filename=f'agents/plot_{model_name}.png')
    model.save(f"agents/model_{model_name}")

    vec_env.close()
