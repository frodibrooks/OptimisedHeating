import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# Set the working directory to where the model is stored
program_dir = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/models"
os.chdir(program_dir)

# === Load environment ===
demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern_2024-11-03"
env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0, demand_pattern=demand_pattern_path, episode_len=24)

# === Load trained model ===
state_dim = int(env.observation_space().shape[0])
action_dim = len(env.action_map)

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_modelVol3.pth"))
model.eval()

# === Run validation ===
full_logs = []
state = env.reset()
max_steps = min(env.episode_len, len(env.demand_pattern))

for timestep in range(max_steps):
    print(f"timestep {timestep + 1}/{max_steps}")
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)
        action_idx = torch.argmax(q_values).item()

    print(f"action_idx: {action_idx}")
    print(f"Q-values: {q_values.numpy()}")

    state, reward, done, info = env.step(action_idx)

    row = {
        "Step": timestep,
        "Action": action_idx,
        "DemandScale": env.demand_pattern[timestep],
        "Reward": reward,
        "EffRatio": env.eff_ratio,
        "ValidHeadsRatio": env.valid_heads_ratio
    }

    # Log Q-values for all actions
    for i, q in enumerate(q_values):
        row[f"Q_{i}"] = q.item()

    # Log pressures
    for junction in env.wds.junctions:
        row[f"Head_{junction.uid}"] = junction.pressure

    # Log pump speeds
    for pump_id, speed in env.pump_speeds.items():
        row[f"PumpSpeed_{pump_id}"] = speed

    # Log pump powers
    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):
        row[f"PumpPower_{pump_id}"] = power

    full_logs.append(row)

    if done:
        break

# Save to CSV
df = pd.DataFrame(full_logs)
csv_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/validation"
os.chdir(csv_path)
df.to_csv("validation_full_log.csv", index=False)

print("Validation complete. Results saved to validation_full_log.csv.")
