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

    def reset(self, demand_pattern=None, training=False):
        self.demand_index = 0
        self.timestep = 0

        # Load new demand pattern if provided
        if demand_pattern is not None:
            self.demand_pattern = self._load_pattern(demand_pattern)

        # Determine scale
        if self.demand_pattern is None and training:
            self.episode_demand_scale = np.random.uniform(0.75, 1.4)
        else:
            self.episode_demand_scale = 1.0

        # === First reset base env ===
        state = super().reset()  # might reinitialize pressures, heads, etc.

        # === Now apply scaled demand for timestep 0 ===
        demand_scale = self.episode_demand_scale
        if not self.use_constant_demand and self.demand_pattern is not None and len(self.demand_pattern) > 0:
            demand_scale *= self.demand_pattern[0]  # timestep 0

        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        self.wds.solve()
        self.pump_power()
        self.calculate_pump_efficiencies()

        return self.get_state()






    def action_index_to_list(self, action_idx):
        """Convert action index to a tuple of speed indices."""
        return self.action_map[action_idx]
    

    def get_state_named(self) -> dict:
        """Return a dictionary of named state values, including pressures, demands, pump speeds, and pump power."""
        self.wds.solve()  # Ensure the WDS is solved before getting state
        state_dict = {}

        # Junction pressures, heads, and demands
        for junction in self.wds.junctions:
            uid = junction.uid
            state_dict[f"{uid}_pressure"] = junction.pressure
            state_dict[f"{uid}_head"] = junction.head
            state_dict[f"{uid}_demand"] = junction.basedemand

        # Pump speeds and power
        for pump_id, speed in self.pump_speeds.items():
            state_dict[f"{pump_id}_speed"] = speed
            state_dict[f"{pump_id}_power"] = self.wds.pumps[pump_id].energy  # Add power per pump

        return state_dict


    def step(self, action_idx):
        speed1, speed2, speed3 = self.action_map[action_idx]

        for group_idx, speed in zip(range(len(self.pumpGroups)), [speed1, speed2, speed3]):
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
        demand = self.get_demand()
        self.timestep += 1
        done = self.timestep >= self.episode_len

        return state,demand, reward, done, {}






def printing_states(step,inp_demand_pattern):
       
    env = WdsWithDemand(demand_pattern=np.array([inp_demand_pattern]))


    # # # # # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    state, demand, reward, done, _ = env.step(step)
    # return step,reward
    # pump_effs = env.pumpEffs
   

    print(f"Reward {reward:.3f} speed {env.pump_speeds}")
    print(f"State {state[:10]}")
    # print(f"Flows: {state[-5:]}")
    print("Total Power:", env.total_power)
    print(f"Energy reward: {float(-env.total_power) / 116:.3f}")

    print(f"Demand: {demand[:6]}")
    print(f"Reward: {reward:.3f}")


    # print(f"Pump speeds: {env.pump_speeds}")
    # print()

    # print(f"Demand Scale: {env.demand_pattern[env.demand_index-1]}")

    # print(f"Pump power: {env.pumpPower}")
    # print()

    # print(f"Valid heads ratio: {env.valid_heads_ratio}")
    # print(f"Pressure Score: {env.pressure_score}")
    # # print(f"Eff ratio: {env.eff_weight*env.eff_ratio}")
    # print(f"Energy reward: {-env.power_penalty_weight*env.total_power}")
    # print(f"Total Energy: {env.total_power}")


    # print(f"Reward: {reward}")
    # print()
    # print(demand[:10])





if __name__ == "__main__":

    # ptr = np.array([1])
    # printing_states(177,ptr)

    # ptr = np.array([1])
    # printing_states(180,ptr)

    # ptr = np.array([1])
    # printing_states(176,ptr)


    # ptr = np.array([0.8])
    # printing_states(195,ptr)

    # ptr = np.array([1])
    # printing_states(195,ptr)

    # ptr = np.array([1])
    # printing_states(198,ptr)   
    # my_dict = {}  
    
    # env = WdsWithDemand(demand_pattern=np.array([0.8]))
    # for i in range(len(env.action_map)):
    #     ptr = np.array([0.8])
    #     step, reward = printing_states(i, ptr)
    #     my_dict[step] = reward
    #     print(f"\rStep {i}/{len(env.action_map)}", end="", flush=True)


    # # Sort the dictionary by reward (values), descending
    # sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))

    # # Print the top 10 entries
    # print("Top 10 actions with highest rewards:")
    # for i, (step, reward) in enumerate(sorted_dict.items()):
    #     if i >= 10:
    #         break
    #     print(f"{step}: {reward}")


    
    # env = WdsWithDemand(demand_pattern=np.array([1]))
    # # print(len(env.action_map))
    # # print(env.action_map[1038])

    # env = WdsWithDemand(demand_pattern=np.array([1]))
    # print(env.action_map[1038])


    ptr = np.array([0.8])
    printing_states(100,ptr)
    printing_states(91,ptr)
    # # printing_states(1477,ptr)



    # ptr = np.array([1])
    # printing_states(1050,ptr)

    # # # print(len(env.wds.junctions) + len(env.pump_speeds)*2 )
    
    # for i in range(800,1200):
    #     print(i, env.action_map[i])

    # env = WdsWithDemand(demand_pattern=np.array([1]))s
    # state = env.reset()
    # print(f"Initial state: {state[:10]}")

    # env = WdsWithDemand(demand_pattern=np.array([0.8, 0.9 , 1.0, 1.1, 1.2, 1.3, 1.4]))
    # # env.reset(demand_pattern=np.array([0.8, 0.9 , 1.0, 1.1, 1.2, 1.3, 1.4]))

    # for i in range(7):
    #     state, demand, reward, done, info = env.step(1477)
    #     print(state[:10],env.demand_pattern[i])