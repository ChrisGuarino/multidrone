from envs.multi_drone_env import MultiDroneEnv

env = MultiDroneEnv(num_drones=2, render_mode="human")

num_episodes = 500
render_every = 10  # render every 10 episodes

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = True
    total_reward = 0

    while done:
        # Sample random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Render every N episodes
        if episode % render_every == 0:
            env.render()

    print(f"Episode {episode} finished with total reward: {total_reward}")

env.close()
