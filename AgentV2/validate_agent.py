import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# Set the program directory and change the working directory
program_dir = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\models"
os.chdir(program_dir)  # Change the current working directory to program_dir

# === Load environment ===
demand_pattern_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\tests\demand_pattern_2024-11-03"
env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0, demand_pattern=demand_pattern_path, episode_len=1)

# === Load trained model ===
state_dim = int(env.observation_space().shape[0])
action_dim = 6  # This needs to match the number of discrete actions (e.g., 3 pump groups * 2 actions per group)

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# === Run validation ===
full_logs = []  # Storage for full logs

# Each timestep corresponds to one hour of the demand pattern
for timestep in range(len(env.demand_pattern)):  # 24 hours
    print(f"timestep {timestep}/24")
    
    # Reset the environment for the specific timestep (hour)
    state = env.reset()
    
    # Get the state tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # Predict the best action based on the current state
    with torch.no_grad():
        q_values = model(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()

    print(f"action_idx: {action_idx}")

    # === Step the environment directly with the integer action ===
    state, reward, done, info = env.step(action_idx)

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

    # Add pump speeds to the log
    for pump_id, speed in env.pump_speeds.items():
        row[f"PumpSpeed_{pump_id}"] = speed
    
    # Ensure you are logging the pump power correctly
    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):  # Ensure you map power to the correct pump
        row[f"PumpPower_{pump_id}"] = power

    # Store the log row
    full_logs.append(row)

# === Save all logs to CSV ===
df = pd.DataFrame(full_logs)
df.to_csv("validation_full_log.csv", index=False)

print("Validation complete. Results saved to validation_full_log.csv.")
