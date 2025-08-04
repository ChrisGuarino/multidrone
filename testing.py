from stable_baselines3 import PPO
from pybullet_env import DroneEnv
from pathlib import Path

def main():
    # Create your environment
    env = DroneEnv(render=True)
    env.max_steps = 1000000

    # Load trained model
    base_dir = Path(__file__).parent
    which_model = input('Which model?: ')
    model_path = base_dir / "agents" / f"model_{which_model}.zip"

    # Convert to string if needed:
    model = PPO.load(str(model_path), env=env)
    
    # Reset environment
    obs, info = env.reset()
    done = False

    while not done:
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f'Timestep: {env.step_counter}', obs, reward, terminated, truncated, {})
    # Cleanup
    env.close()

if __name__ == "__main__":
    main() 
