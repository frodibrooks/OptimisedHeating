import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# === Set working directory to where model is stored ===
program_dir = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/models"
os.chdir(program_dir)

# === Load environment with extended episode ===
demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern_2024-11-03"
env = WdsWithDemand(
    eff_weight=3.0,
    pressure_weight=1.0,
    demand_pattern=demand_pattern_path,
    episode_len=100  # Agent can take 100 actions
)

# === Load trained model ===
state_dim = int(env.observation_space().shape[0])
action_dim = len(env.action_map)

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model_vol4.pth"))
model.eval()

# === Run validation ===
full_logs = []
state = env.reset()

for timestep in range(env.episode_len):
    print(f"timestep {timestep + 1}/{env.episode_len}")
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

    for i, q in enumerate(q_values):
        row[f"Q_{i}"] = q.item()

    for junction in env.wds.junctions:
        row[f"Head_{junction.uid}"] = junction.pressure

    for pump_id, speed in env.pump_speeds.items():
        row[f"PumpSpeed_{pump_id}"] = speed

    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):
        row[f"PumpPower_{pump_id}"] = power

    full_logs.append(row)

    if done:
        print("Episode terminated early.")
        break

# === Save to CSV ===
df = pd.DataFrame(full_logs)
csv_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/validation"
os.chdir(csv_path)
df.to_csv("validation_full_log_agent4.csv", index=False)

print("Validation complete. Results saved to validation_full_log_agent4.csv.")
