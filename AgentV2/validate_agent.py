import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# === Set paths ===
# program_dir = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/models"
program_dir = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\models"
demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern_2024-11-03"
# demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern"
# save_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/validation"
save_path = r"C:\Users\frodi\Documents\OptimisedHeating\validation"


# === Load environment ===
os.chdir(program_dir)

# env = WdsWithDemand(
#     eff_weight=3.0,
#     pressure_weight=1,
#     demand_pattern=demand_pattern_path,
#     episode_len = 24, # Þetta er lengd demand pattern
#     use_constant_demand=False
# )

# demand_ptr = np.array([0.8 , 0.9 , 1, 1.1, 1.2 , 1.3])
demand_ptr = np.array([1,1,1,1,1])
env = WdsWithDemand(
    demand_pattern=demand_ptr, # Þetta er demand pattern
    episode_len = len(demand_ptr) ,# Þetta er lengd demand pattern
    use_constant_demand=False

)

# === Load model ===
state_dim = int(env.observation_space().shape[0])
action_dim = len(env.action_map)


model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model_vol101.pth"))
model.eval()

# === Run validation ===
full_logs = []
env.reset(demand_pattern=demand_ptr)

state = env.get_state()

for timestep in range(env.episode_len):

    demand = env.get_demand()
    print(f"Agent sees state {state[-8:]}")
    print(f"Agent sees demand {state[:10]}")

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)
        action_idx = torch.argmax(q_values).item()


    print(f"Pick speed {env.action_map[action_idx]}")
    state, reward, done, info = env.step(action_idx)
    print(f"New state {state[-8:]}")


    print(f"timestep {timestep + 1}/{env.episode_len}")
    print(f"Agent sees demands with scaling: {env.demand_pattern[timestep]:.2f}")  # or use env.demand_pattern[timestep+1] safely
    print()
    print(f"Reward: {reward:.3f}")
    print()
    print(env.demand_pattern[timestep])

    # print("Q-values at timestep 1:", q_values.tolist())

    

    # === Log everything ===
    row = {
        "Step": timestep,
        "ActionIndex": action_idx,
        "DemandScale": env.demand_pattern[timestep],
        "Reward": reward,
        "EffReward": env.eff_ratio*env.eff_weight,
        "Valid heads ratio": env.valid_heads_ratio,
        "Energy reward": -env.total_power*env.power_penalty_weight,
    }

    # Flatten q_values to 1D tensor to avoid multi-element tensors during iteration
    q_values_flat = q_values.flatten()

    for i, q in enumerate(q_values_flat):
        row[f"Q_{i}"] = q.item()


    # Log pressures
    for junction in env.wds.junctions:
        row[f"Head_{junction.uid}"] = junction.pressure
    # log Demand
    for junction in env.wds.junctions:
        row[f"Demand_{junction.uid}"] = junction.basedemand

    # Log actual speeds from action map
    speed1, speed2 = env.action_map[action_idx]

    row["PumpGroupSpeed_1"] = speed1
    row["PumpGroupSpeed_2"] = speed2

    # Log pump powers
    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):
        row[f"PumpPower_{pump_id}"] = power

    full_logs.append(row)



# === Save logs ===
df = pd.DataFrame(full_logs)
os.chdir(save_path)
df.to_csv("validation_full_log_agent101.csv", index=False)

print("Validation complete. Results saved to validation_full_log_agent101.csv.")
