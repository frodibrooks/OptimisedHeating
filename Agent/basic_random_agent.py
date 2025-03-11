
import numpy as np
from pump_env import PumpEnv  # Import the PumpEnv class

# Create the environment
env = PumpEnv()

# Run a test episode with random actions
num_episodes = 5
num_steps = 20  # Adjust if needed

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    print(f"Episode {episode + 1}:")
    
    for step in range(num_steps):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Action: {action}, Reward: {reward:.2f}")

        if done:
            break

    print(f"Total reward: {total_reward:.2f}\n")

env.close()
