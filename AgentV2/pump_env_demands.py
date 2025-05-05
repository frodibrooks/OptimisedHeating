import pandas as pd
import numpy as np
from pump_env import wds

class WdsWithDemand(wds):
    def __init__(self, demand_pattern=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demand_pattern = self._load_pattern(demand_pattern)
        self.demand_index = 0

    def _load_pattern(self, pattern):
        """Load a demand pattern from a CSV or use the provided numpy array."""
        if isinstance(pattern, str):
            return pd.read_csv(pattern)['demand_pattern'].values
        elif isinstance(pattern, np.ndarray):
            return pattern
        else:
            return None

    def reset(self, demand_pattern=None):
        """Reset the environment with a given or existing demand pattern."""
        self.demand_index = 0
        if demand_pattern is not None:
            self.demand_pattern = self._load_pattern(demand_pattern)
        return super().reset()

    def action_index_to_list(self, action_idx):
        """Convert action index to a tuple of speed indices."""
        return self.action_map[action_idx]

    def step(self, action_idx):
        # Extract idx1, idx2 from the action_idx
        idx1, idx2 = self.action_index_to_list(action_idx)
        
        # Ensure idx1 and idx2 are valid indices for speed_levels
        if not (0 <= idx1 < len(self.speed_levels)) or not (0 <= idx2 < len(self.speed_levels)):
            raise ValueError(f"Invalid indices: idx1={idx1}, idx2={idx2} for speed_levels.")

        # Get pump speeds based on the action
        speed1 = self.speed_levels[idx1]
        speed2 = self.speed_levels[idx2]

        # Update pump speeds
        for group_idx, speed in zip(range(len(self.pumpGroups)), [speed1, speed2]):
            for pump_id in self.pumpGroups[group_idx]:
                self.pump_speeds[pump_id] = speed
                self.wds.pumps[pump_id].speed = speed

        # Apply random demand scaling factor per episode
        demand_scale = np.random.uniform(0.8, 1.2)

        # Temporal variation by modifying demand scaling per step
        if self.demand_pattern is not None and self.demand_index < len(self.demand_pattern):
            demand_scale *= self.demand_pattern[self.demand_index]
            self.demand_index += 1

        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        # Simulate and compute reward
        self.wds.solve()
        self.pump_power()
        self.calculate_pump_efficiencies()
        reward = self._compute_reward()
        state = self.get_state()
        self.timestep += 1
        done = self.timestep >= self.episode_len

        return state, reward, done, {}




if __name__ == "__main__":
        
    SPEED_LEVELS = np.round(np.arange(0.8, 1.301, 0.025), 3)
    ACTION_MAP = [(s1, s2) for s1 in SPEED_LEVELS for s2 in SPEED_LEVELS]
   
    env = WdsWithDemand(action_map=ACTION_MAP, eff_weight=3.0, pressure_weight=1.0, episode_len=300)

    # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(168)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Pump efficiencies: {env.pumpEffs}")
    print()

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Eff ratio: {3*env.eff_ratio}")
    print(f"Energy: {-0.01*env.total_power}")

    print(f"Reward: {reward}")
    print()

    


    print(f"States: {states[:10]}")

    env.step(176)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Pump efficiencies: {env.pumpEffs}")
    print()

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Reward: {reward}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Eff ratio: {3*env.eff_ratio}")


    print(f"States: {states[:10]}")
    # for i in range(len(ACTION_MAP)):
    #     print(f"Action {i}: {ACTION_MAP[i]}")
