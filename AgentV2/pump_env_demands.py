import pandas as pd
import numpy as np
from pump_env import wds

class WdsWithDemand(wds):
    def __init__(self, demand_pattern=None,use_constant_demand=False ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demand_pattern = self._load_pattern(demand_pattern)
        self.demand_index = 0
        self.episode_demand_scale = 1.0
        self.use_constant_demand = use_constant_demand


    def _load_pattern(self, pattern):
        """Load a demand pattern from a CSV or use the provided numpy array."""
        if isinstance(pattern, str):
            return pd.read_csv(pattern)['demand_pattern'].values
        elif isinstance(pattern, np.ndarray):
            return pattern
        else:
            return None

    def reset(self, demand_pattern=None, training = False):
        self.demand_index = 0
        
        # If using a pattern, base scale is 1.0
        if self.demand_pattern is not None:
            self.episode_demand_scale = 1.0
        if training:
            # Use random demand scale if no pattern is provided
            self.episode_demand_scale = np.random.uniform(0.75, 1.4)

        if demand_pattern is not None:
            self.demand_pattern = self._load_pattern(demand_pattern)

        return super().reset()




    def action_index_to_list(self, action_idx):
        """Convert action index to a tuple of speed indices."""
        return self.action_map[action_idx]

    def step(self, action_idx):
        speed1, speed2 = self.action_map[action_idx]

        for group_idx, speed in zip(range(len(self.pumpGroups)), [speed1, speed2]):
            for pump_id in self.pumpGroups[group_idx]:
                self.pump_speeds[pump_id] = speed
                self.wds.pumps[pump_id].speed = speed

        # Default scale
        demand_scale = self.episode_demand_scale

        if not self.use_constant_demand and self.demand_pattern is not None and self.demand_index < len(self.demand_pattern):
            demand_scale *= self.demand_pattern[self.demand_index]
            # print(f"[STEP] timestep={self.demand_index}, demand_multiplier={self.demand_pattern[self.demand_index]}")
            self.demand_index += 1

        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        self.wds.solve()
        self.pump_power()
        self.calculate_pump_efficiencies()
        reward = self._compute_reward()
        state = self.get_state()
        self.timestep += 1
        done = self.timestep >= self.episode_len

        return state, reward, done, {}






if __name__ == "__main__":

   
    env = WdsWithDemand(demand_pattern=np.array([1.3]))

    # # # # # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(177)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Demand Scale: {env.demand_pattern[env.demand_index-1]}")

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Eff ratio: {env.eff_weight*env.eff_ratio}")
    print(f"Energy reward: {-env.power_penalty_weight*env.total_power}")
    print(f"Total Energy: {env.total_power}")


    print(f"Reward: {reward}")
    print()

    env = WdsWithDemand(demand_pattern=np.array([1.3]))

    # # # # # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(176)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Demand Scale: {env.demand_pattern[env.demand_index-1]}")

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Eff ratio: {env.eff_weight*env.eff_ratio}")
    print(f"Energy reward: {-env.power_penalty_weight*env.total_power}")
    print(f"Total Energy: {env.total_power}")


    print(f"Reward: {reward}")
    print()


  



    

    
    for i in range(len(env.action_map)):
        print(f"Action {i}: {env.action_map[i]}")
  

  