import gymnasium as gym
from stable_baselines3 import PPO
from env import DroneEnv  # Import your environment
import time

start = time.time()

# Create the environment
env = DroneEnv(render=False)

# Create PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.0,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10
)

# Train
model.learn(total_timesteps=200_000)

end = time.time() 
print(f'Time elapsed: {end-start:.2f} seconds') 

model_name = input("Enter model name: ")
# plot_list(env.reward_track, filename=f'agents/plot_{model_name}.png')
model.save(f"agents/model_{model_name}")

env.close()
