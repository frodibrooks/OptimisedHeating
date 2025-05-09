import pandas as pd
import numpy as np
from pump_env import wds

class WdsWithDemand(wds):
    def __init__(self, demand_pattern=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demand_pattern = self._load_pattern(demand_pattern)
        self.demand_index = 0
        self.episode_demand_scale = 1.0


    def _load_pattern(self, pattern):
        """Load a demand pattern from a CSV or use the provided numpy array."""
        if isinstance(pattern, str):
            return pd.read_csv(pattern)['demand_pattern'].values
        elif isinstance(pattern, np.ndarray):
            return pattern
        else:
            return None

    def reset(self, demand_pattern=None):
        self.demand_index = 0
        self.episode_demand_scale = np.random.uniform(0.8, 1.2)  # One scale per episode
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

        demand_scale = 1

        # Optional: combine with temporal pattern if provided
        if self.demand_pattern is not None and self.demand_index < len(self.demand_pattern):
            demand_scale *= self.demand_pattern[self.demand_index]
            self.demand_index += 1
        self.episode_demand_scale = demand_scale
        
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

   
    env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0,demand_pattern=np.array([1.045780954728453]))

    # # # # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(124)
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
    print(f"Energy: {-0.02*env.total_power}")

    print(f"Reward: {reward}")
    print()

    
    env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0,demand_pattern=np.array([1.045780954728453]))


    env.step(116)
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
    print(f"Energy: {-0.02*env.total_power}")

    print(f"Reward: {reward}")
    

    env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0,demand_pattern=np.array([1.045780954728453]))


    env.step(88)
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
    print(f"Energy: {-0.02*env.total_power}")

    print(f"Reward: {reward}")
    print()



    
  

    
    # for i in range(len(env.action_map)):
    #     print(f"Action {i}: {env.action_map[i]}")
  
    # # print(f"Action map: {env.action_map}")
    # state_dim = int(env.observation_space().shape[0])
    # action_dim = len(env.action_map)
    # print(state_dim,action_dim)
   
  