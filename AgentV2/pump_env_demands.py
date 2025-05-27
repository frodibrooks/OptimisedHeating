import pandas as pd
import numpy as np
from pump_env import wds

class WdsWithDemand(wds):
    def __init__(self, demand_pattern=None, use_constant_demand=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demand_pattern = self._load_pattern(demand_pattern)
        self.demand_index = 0
        self.episode_demand_scale = 1.0
        self.use_constant_demand = use_constant_demand
        self.timestep = 0

    def _load_pattern(self, pattern):
        """Load a demand pattern from CSV or numpy array, or None."""
        if isinstance(pattern, str):
            df = pd.read_csv(pattern)
            # Assumes pattern column named 'demand_pattern' or adapt accordingly
            return df['demand_pattern'].values  
        elif isinstance(pattern, np.ndarray):
            return pattern.flatten()  # flatten in case of 2D array like np.array([[...]])
        else:
            return None

    def scale_demands(self, scale):
        """Apply given demand scale to all junctions based on base demands."""
        for junction in self.wds.junctions:
            base = self.demandDict[junction.uid]
            junction.basedemand = base * scale

    def reset(self, demand_pattern=None, randomize_demand=False):
        """
        Reset environment.
        
        Parameters:
            demand_pattern: Optional new demand pattern to load.
            randomize_demand: If True, pick a random base demand scale; else fixed 1.0.
        """
        self.timestep = 0
        self.demand_index = 0

        if demand_pattern is not None:
            self.demand_pattern = self._load_pattern(demand_pattern)

        # Decide base episode demand scale
        if randomize_demand:
            self.episode_demand_scale = np.random.uniform(0.75, 1.4)
        else:
            self.episode_demand_scale = 1.0

        # Compose initial demand scale
        demand_scale = self.episode_demand_scale
        if not self.use_constant_demand and self.demand_pattern is not None and len(self.demand_pattern) > 0:
            demand_scale *= self.demand_pattern[0]
            self.demand_index = 1  # step() will continue from next index

        self.scale_demands(demand_scale)

        # Call base reset
        state = super().reset()

        # Solve hydraulic state and calculate metrics
        self.wds.solve()
        self.pump_power()
        self.calculate_pump_efficiencies()

        return self.get_state()

    def step(self, action_idx, training=False):
        """
        Take an action and update environment state.
        
        Parameters:
            action_idx: Discrete action index.
            training: If True, randomize demand scale at each step; else use demand pattern.
        """
        speed1, speed2 = self.action_map[action_idx]

        # Apply pump speeds
        for group_idx, speed in enumerate([speed1, speed2]):
            for pump_id in self.pumpGroups[group_idx]:
                self.pump_speeds[pump_id] = speed
                self.wds.pumps[pump_id].speed = speed

        # Determine demand scale for this step
        if self.use_constant_demand:
            demand_scale = 1.0
        else:
            if training:
                # Randomize demand scale every step during training
                demand_scale = np.random.uniform(0.75, 1.4)
            else:
                # Use next value from demand pattern, or repeat last if at end
                if self.demand_pattern is not None and self.demand_index < len(self.demand_pattern):
                    demand_scale = self.episode_demand_scale * self.demand_pattern[self.demand_index]
                    self.demand_index += 1
                else:
                    demand_scale = self.episode_demand_scale

        # print(f"[DEBUG] Applying demand_scale {demand_scale:.3f} at timestep {self.timestep}")

        self.scale_demands(demand_scale)

        # Solve hydraulic model and compute metrics
        self.wds.solve()
        self.pump_power()
        self.calculate_pump_efficiencies()

        reward = self._compute_reward()
        state = self.get_state()
        self.timestep += 1
        done = self.timestep >= self.episode_len

        return state, reward, done, {}

    def action_index_to_list(self, action_idx):
        return self.action_map[action_idx]




def printing_states(step,inp_demand_pattern):
       
    env = WdsWithDemand(demand_pattern=np.array([inp_demand_pattern]))


    # # # # # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(step)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Demand Scale: {env.demand_pattern[env.demand_index-1]}")

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Pressure Score: {env.pressure_score}")
    # print(f"Eff ratio: {env.eff_weight*env.eff_ratio}")
    print(f"Energy reward: {-env.power_penalty_weight*env.total_power}")
    print(f"Total Energy: {env.total_power}")


    print(f"Reward: {reward}")
    print()





if __name__ == "__main__":

    # ptr = np.array([1])
    # printing_states(218,ptr)

    # ptr = np.array([1])
    # printing_states(195,ptr)

    ptr = np.array([0.9])
    printing_states(182,ptr)
  


    
    # env = WdsWithDemand(demand_pattern=np.array([1]))
    
    # for i in range(len(env.action_map)):
    #     print(f"Action {i}: {env.action_map[i]}")