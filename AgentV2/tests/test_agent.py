# test_agent.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from pump_env import wds
from training import DQN


# === Load Environment ===
env = wds(eff_weight=3.0, pressure_weight=1.0)
initial_state = env.reset()
state_size = len(initial_state)
action_size = 2 * 3  # 3 actions per group, 2 groups

# === Load Trained Model ===
policy_net = DQN(state_size, action_size)
model_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\models\trained_model.pth"
policy_net.load_state_dict(torch.load(model_path))
policy_net.eval()

# === Run Agent ===
total_reward = 0
state = env.reset()

for t in range(env.episode_len):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state_tensor)
        action = torch.argmax(q_values).item()

    # Convert flat action index to multi-action
    group1 = action % 3
    group2 = (action // 3) % 3
    multi_action = [group1, group2]

    next_state, reward, done, _ = env.step(multi_action)
    state = next_state
    total_reward += reward

    print(f"Step {t+1}: Action {multi_action}, Reward {reward:.3f}")

print(f"\nTotal reward: {total_reward:.3f}")
