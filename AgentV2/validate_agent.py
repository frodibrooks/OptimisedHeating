import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# === Set paths ===
program_dir = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/models"
save_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/validation"

# === Demand pattern ===
# demand_ptr = np.array([1, 1.2 , 1.4, 1.2, 1, 0.8, 0.7, 0.9])
demand_ptr = np.array([1.2, 1.2, 1.2, 1.2])

episode_len = len(demand_ptr)

# === Load environment ===
os.chdir(program_dir)

env = WdsWithDemand(
    demand_pattern=demand_ptr,
    episode_len=episode_len,
    use_constant_demand=False
)

# === Load model ===
state_dim = int(env.observation_space().shape[0])
action_dim = len(env.action_map)

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model_vol30.pth"))
model.eval()

# === Run validation ===
full_logs = []
state = env.reset(demand_pattern=demand_ptr)

for timestep in range(env.episode_len):
    # === Agent selects action ===
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)
        action_idx = torch.argmax(q_values).item()

    # === Step environment ===
    next_state, reward, done, info = env.step(action_idx)

    # === Debug print ===
    print(f"Timestep {timestep + 1}/{env.episode_len}")
    print(f"Demand scale: {env.demand_pattern[timestep]:.2f}")
    print(f"Selected pump speeds: {env.action_map[action_idx]}")
    print(f"Reward: {reward:.3f}")
    print(f"Energy: {-env.total_power * env.power_penalty_weight:.3f}\n")

    # === Log row ===
    row = {
        "Step": timestep,
        "ActionIndex": action_idx,
        "DemandScale": env.demand_pattern[timestep],
        "Reward": reward,
        "EffReward": env.eff_ratio * env.eff_weight,
        "Valid heads ratio": env.valid_heads_ratio,
        "Energy reward": -env.total_power * env.power_penalty_weight,
    }

    for i, q in enumerate(q_values):
        row[f"Q_{i}"] = q.item()

    # Junction pressures and demands
    for junction in env.wds.junctions:
        row[f"Head_{junction.uid}"] = junction.pressure
        row[f"Demand_{junction.uid}"] = junction.basedemand

    # Pump speeds
    speed1, speed2 = env.action_map[action_idx]
    row["PumpGroupSpeed_1"] = speed1
    row["PumpGroupSpeed_2"] = speed2

    # Pump powers
    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):
        row[f"PumpPower_{pump_id}"] = power

    full_logs.append(row)

    # Move to next state
    state = next_state

# === Save logs ===
df = pd.DataFrame(full_logs)
os.chdir(save_path)
df.to_csv("validation_full_log_agent30.csv", index=False)

print("Validation complete. Results saved to validation_full_log_agent30.csv.")
