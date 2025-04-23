import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env import WdsWithDemand

# === Load environment ===
demand_pattern_path = "tests/demand_pattern_2024-11-03.csv"
env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0, demand_pattern=demand_pattern_path, episode_len=1)

# === Load trained model ===
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# === Run validation ===
full_logs = []  # Storage for full logs

# Each timestep corresponds to one hour of the demand pattern
for timestep in range(len(env.demand_pattern)):  # 24 hours
    print(f"timestep {timestep}/24")
    # Reset the environment for the specific timestep (hour)
    state = env.reset(timestep=timestep)
    
    # Get the state tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # Predict the best action based on the current state
    with torch.no_grad():
        q_values = model(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()

    action = env.discrete_to_action(action_idx)  # Convert to action if necessary
    state, reward, done, info = env.step(action)  # Take action in the environment

    # === Log system state ===
    row = {
        "Step": timestep,  # Log the current timestep (hour)
        "Action": action_idx,
        "DemandScale": env.demand_pattern[timestep],  # Current demand scale
        "Reward": reward,
    }

    # Add junction heads to the log
    for junction in env.wds.junctions:
        row[f"Head_{junction.uid}"] = junction.head

    # Add pump speeds and power to the log
    for pump_id, speed in env.pump_speeds.items():
        row[f"PumpSpeed_{pump_id}"] = speed
    for pump_id, power in env.pump_power.items():
        row[f"PumpPower_{pump_id}"] = power

    # Store the log row
    full_logs.append(row)

# === Save all logs to CSV ===
df = pd.DataFrame(full_logs)
df.to_csv("validation_full_log.csv", index=False)

print("Validation complete. Results saved to validation_full_log.csv.")
