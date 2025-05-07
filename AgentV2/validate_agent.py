import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# === Set paths ===
program_dir = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/models"
demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern_2024-11-03"
save_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/validation"


# === Load environment ===
os.chdir(program_dir)
env = WdsWithDemand(
    eff_weight=3.0,
    pressure_weight=1.0,
    demand_pattern=demand_pattern_path,
    episode_len = 23 # Þetta er lengd demand pattern

)

# === Load model ===
state_dim = int(env.observation_space().shape[0])
action_dim = len(env.action_map)


model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model_vol11.pth"))
model.eval()

# === Run validation ===
full_logs = []
state = env.get_state()

for timestep in range(env.episode_len):
    print(f"timestep {timestep + 1}/{env.episode_len}")
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)
        action_idx = torch.argmax(q_values).item()

    print(f"action_idx: {action_idx} Speeds: {env.action_map[action_idx]}")
    # print(f"Q-values: {q_values.numpy()}")

    state, reward, done, info = env.step(action_idx)

    # === Log everything ===
    row = {
        "Step": timestep,
        "ActionIndex": action_idx,
        "DemandScale": env.demand_pattern[timestep],
        "Reward": reward,
        "EffRatio": env.eff_ratio,
        "ValidHeadsRatio": env.valid_heads_ratio,
        "Energy": sum(env.pumpPower),
    }

    # Log Q-values
    for i, q in enumerate(q_values):
        row[f"Q_{i}"] = q.item()

    # Log pressures
    for junction in env.wds.junctions:
        row[f"Head_{junction.uid}"] = junction.pressure

    # Log actual speeds from action map
    speed1, speed2 = env.action_map[action_idx]

    row["PumpGroupSpeed_1"] = speed1
    row["PumpGroupSpeed_2"] = speed2

    # Log pump powers
    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):
        row[f"PumpPower_{pump_id}"] = power

    full_logs.append(row)

    if done:
        print("Episode terminated early.")
        break

# === Save logs ===
df = pd.DataFrame(full_logs)
os.chdir(save_path)
df.to_csv("validation_full_log_agent11.csv", index=False)

print("Validation complete. Results saved to validation_full_log_agent11.csv.")
