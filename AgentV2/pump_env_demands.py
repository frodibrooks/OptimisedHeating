import pandas as pd
import numpy as np
from pump_env import wds

class WdsWithDemand(wds):
    def __init__(self, demand_pattern=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If demand pattern is provided, load it
        if demand_pattern is not None:
            self.demand_pattern = pd.read_csv(demand_pattern)['demand_pattern'].values
        else:
            self.demand_pattern = None

        self.demand_index = 0  # To track where we are in the demand pattern

    def reset(self):
        # Reset base class functionality
        state = super().reset()
        
        # Reset demand index for new episode
        self.demand_index = 0

        return state

    def step(self, action):
        # Execute action using base class method
        state, reward, done, info = super().step(action)
        
        # Apply the demand pattern if available
        if self.demand_pattern is not None and self.demand_index < len(self.demand_pattern):
            demand_scale = self.demand_pattern[self.demand_index]  # Get the current demand scaling factor
            self.demand_index += 1
        else:
            demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)

        # Update demands in the network with the new demand scale
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        return state, reward, done, info




# Set the path to your demand pattern CSV
demand_pattern_path = r'C:\Users\frodi\Documents\OptimisedHeating\AgentV2\tests\demand_pattern_2024-11-03'

# Initialize the environment with the demand pattern
env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0, demand_pattern=demand_pattern_path)

# Reset the environment
state = env.reset()

# Print initial state and demand pattern
print("Initial State:", state[:10])
print("\nDemand Pattern (Scaling Factor):")
print(env.demand_pattern)  # Print the entire demand pattern

# Run through the demand pattern and print out the demand at each step
for t in range(len(env.demand_pattern)):
    # Print demand at current timestep
    demand_scale = env.demand_pattern[t]
    print(f"Step {t+1}: Applied Demand Scale = {demand_scale}")

    # Apply the demand change in the environment
    state, reward, done, info = env.step([1, 1])  # Using a dummy action as we just want to test demand change

    # Print the updated demands at each junction
    print(f"Junction Demands at Step {t+1}:")
    for junction in list(env.wds.junctions)[:5]:
        print(f"  {junction.uid}: {junction.basedemand}")

    # Optionally print the pump speeds or other relevant system states
    print(f"Pump Speeds at Step {t+1}:")
    for pump_id, speed in env.pump_speeds.items():
        print(f"  Pump {pump_id}: Speed {speed:.3f}")

    # Print the reward at this step for debugging purposes
    print(f"Reward at Step {t+1}: {reward:.3f}")
    
    print("-" * 50)