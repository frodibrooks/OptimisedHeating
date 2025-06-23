import torch
import pandas as pd
import numpy as np
from training import DQN
from pump_env_demands import WdsWithDemand
import os

# === Set paths ===
program_dir = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/models"
demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern_2024-11-03"
# demand_pattern_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern"
save_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/validation/logs"


# === Load environment ===
os.chdir(program_dir)



# demand_ptr = np.array([0.8,0.9,1.0,1.1,1.2,1.3,1.4])
# demand_ptr = np.array([1, 0.8, 0.8, 0.8,0.8, 0.8])
# demand_ptr = np.array([1.3,1.3,1.3,1.4,1.4,1.4])


# demand_ptr = np.array([
#     0.896, 0.897, 0.896, 0.899, 0.921, 0.969,
#     1.087, 1.031, 1.047, 1.052, 1.056, 1.055,
#     1.050, 1.038, 0.993, 0.989, 1.064, 1.102,
#     1.161, 1.132, 1.157, 1.122, 1.080
# ])   # haust dagur goðar nidustoduer

demand_ptr = np.array([
    0.928, 0.927, 0.920, 0.900, 0.895, 0.932,
    1.011, 0.987, 0.993, 1.002, 0.998, 1.002,
    0.990, 0.990, 0.990, 0.993,
    1.046, 1.088, 1.120, 1.119, 1.080, 1.056, 1.032
])


# demand_ptr = np.array([
#     1.140, 1.179, 1.159, 1.159, 1.146, 1.186,
#     1.268, 1.287, 1.315, 1.315, 1.307, 1.270,
#     1.253, 1.249, 1.249, 1.232, 1.267, 1.335,
#     1.40, 1.383, 1.389, 1.357, 1.308
# ])


env = WdsWithDemand(
    demand_pattern=demand_ptr, # Þetta er demand pattern
    episode_len = len(demand_ptr) ,# Þetta er lengd demand pattern
    use_constant_demand=False

)

# === Load model ===
state_dim = int(env.observation_space().shape[0])
action_dim = len(env.action_map)


model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("trained_model_vol406.pth"))
model.eval()

# === Run validation ===
full_logs = []
env.reset(demand_pattern=demand_ptr)

state = env.get_state()

for timestep in range(env.episode_len):

    # print(f"Agent sees state: {state[:10]}")
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)
        action_idx = torch.argmax(q_values).item()



    state, demand, reward, done, info = env.step(action_idx)



    print(f"timestep {timestep + 1}/{env.episode_len}")
    # print(f"Agent sees demands with scaling: {demand[:5]}", env.demand_pattern[timestep])  # or use env.demand_pattern[timestep+1] safely
    print(f"Demand scale: {env.demand_pattern[timestep]:.3f}")
    print(f"Agent selects Speeds: {env.action_map[action_idx]}")
    # print(f"New state: {state[:10]}")
    print()
    print(f"Reward: {reward}")
    print()
    print(f"Energy: {-env.total_power*env.power_penalty_weight:.3f}")

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


    # Log actual speeds from action map
    speed1, speed2, speed3  = env.action_map[action_idx]

    row["PumpGroupSpeed_1"] = speed1
    row["PumpGroupSpeed_2"] = speed2
    row["PumpGroupSpeed_3"] = speed3

    # Log pump powers
    for pump_id, power in zip(env.wds.pumps.keys(), env.pumpPower):
        row[f"PumpPower_{pump_id}"] = power

    full_logs.append(row)



# # === Save logs ===
df = pd.DataFrame(full_logs)
os.chdir(save_path)
df.to_csv("validation_full_log_agent406_spring.csv", index=False)

print("Validation complete. Results saved to validation_full_log_agent406.csv.")