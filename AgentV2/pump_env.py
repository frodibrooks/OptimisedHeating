import os
import numpy as np
import gym
from epynet import Network
from ecurves import eff_curves

class wds():
    def __init__(self, 
                 wds_name="Vatnsendi",
                 speed_increment=0.025,
                 episode_len=300,
                 pump_groups=[['17', '10','25', '26'], ['27']],
                 total_demand_lo=0.8,
                 total_demand_hi=1.2,
                 seed=None,
                 eff_weight=1.0,
                 pressure_weight=1.0,
                 random_demand_scaling=False,
                 demand_scale=None):
        if seed:
            np.random.seed(seed)

        pathToRoot = os.path.dirname(os.path.realpath(__file__))
        pathToWDS = os.path.join(pathToRoot, "water_network", wds_name + ".inp")
        self.wds = Network(pathToWDS)

        self.demandDict = self.build_demand_dict()
        self.pumpGroups = pump_groups

        self.pump_speeds = {pid: 1.0 for group in pump_groups for pid in group}
        self.pumpEffs = {pid: 1.0 for group in pump_groups for pid in group}

        self.episode_len = episode_len
        self.total_demand_lo = total_demand_lo
        self.total_demand_hi = total_demand_hi
        self.speed_increment = speed_increment

        self.headLimitLo = 35
        self.peakTotEff = 0.0629
        self.eff_weight = eff_weight
        self.pressure_weight = pressure_weight
        self.random_demand_scaling = random_demand_scaling
        self.demand_scale = demand_scale

        self.min_speed = 0.7
        self.max_speed = 1.4

        self.eff_ratio = 0.0
        self.valid_heads_ratio = 0.0
        self.timestep = 0

        # Define the speed levels
        self.speed_levels = np.round(np.arange(0.6, 1.251, 0.05), 3)

        # Define the action map
        self.action_map = [
            (s1, s2) 
            for s1 in self.speed_levels 
            for s2 in self.speed_levels
        ]

    def build_demand_dict(self):
        return {j.uid: j.basedemand for j in self.wds.junctions}

    def pump_power(self):
        self.pumpPower = [p.energy for p in self.wds.pumps.values()]

    def reset(self):
        self.timestep = 0
        self.pump_speeds = {pid: 1.0 for group in self.pumpGroups for pid in group}
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid]
        self.wds.solve()
        return self.get_state()

    def get_state(self):
        self.wds.solve()
        pump_speeds = list(self.pump_speeds.values())
        pressures = [j.pressure for j in self.wds.junctions]
        flows = [p.flow for p in self.wds.pumps.values()]
        power = self.pumpPower if hasattr(self, 'pumpPower') else [0.0] * len(self.wds.pumps)
        return pump_speeds + pressures + flows + power

    def step(self, action_idx):
        # Map the action index to pump speeds using the action_map
        idx1, idx2 = self.action_map[action_idx]

        # Apply the chosen speeds to the pumps
        for group_idx, speed in zip(range(len(self.pumpGroups)), [idx1, idx2]):
            for pump_id in self.pumpGroups[group_idx]:
                self.pump_speeds[pump_id] = speed
                self.wds.pumps[pump_id].speed = speed

        if self.random_demand_scaling:
            demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)
            for junction in self.wds.junctions:
                junction.basedemand = self.demandDict[junction.uid] * demand_scale
        elif self.demand_scale is not None:
            demand_scale = self.demand_scale
            for junction in self.wds.junctions:
                junction.basedemand = self.demandDict[junction.uid] * demand_scale
        else:
            demand_scale = 1.0  # No scaling
        # Set the demand scale for the current timestep
        

        # Solve the water distribution system
        self.wds.solve()

        # Calculate pump power and compute the reward
        self.pump_power()
        self.calculate_pump_efficiencies()
        reward = self._compute_reward()

        # Get the next state
        state = self.get_state()

        # Increment the timestep
        self.timestep += 1

        # Check if the episode is done
        done = self.timestep >= self.episode_len

        return state, reward, done, {}

    def _compute_reward(self):
        pump_eff_ok = all(0 <= eff <= 1.2 for eff in self.pumpEffs.values())
        if not pump_eff_ok:
            self.eff_ratio = 0.0
            self.valid_heads_ratio = 0.0
            return 0.0

        pressures = [j.pressure for j in self.wds.junctions]

        # if any(p < 15 or p > 105 for p in pressures):
        #     return 0.0

        group_eff_ratios = {
            i: np.clip(np.mean([self.pumpEffs.get(pid, 0) for pid in group]), 0, 1)
            for i, group in enumerate(self.pumpGroups)
        }
        self.eff_ratio = np.mean(list(group_eff_ratios.values()))
        heads = np.array(pressures)
        self.valid_heads_ratio = np.mean(heads >= self.headLimitLo)

        # Power penalty term
        self.total_power = np.sum(self.pumpPower)
        power_penalty_weight = 0.02 # You can tune this hyperparameter

        self.pressure_score = self.valid_heads_ratio**2

        reward = (
            self.eff_weight * self.eff_ratio
            + self.pressure_weight * self.pressure_score
            - power_penalty_weight * self.total_power
        )
        return reward

    def calculate_pump_efficiencies(self):
        self.pumpEffs = {}
        for group in self.pumpGroups:
            for pump_id in group:
                pump = self.wds.pumps[pump_id]
                try:
                    flow_lps = pump.flow
                    eff = self.calculate_efficiency(pump_id, flow_lps)
                    self.pumpEffs[pump_id] = eff
                except Exception as e:
                    print(f"Efficiency error for pump {pump_id}: {e}")
                    self.pumpEffs[pump_id] = 0

    def calculate_efficiency(self, pump_id, flow_lps):
        curve = eff_curves.get(pump_id)
        if curve:
            try:
                return float(curve(flow_lps))
            except Exception as e:
                print(f"Error calculating efficiency for pump {pump_id}, flow={flow_lps}: {e}")
        else:
            raise ValueError(f"No efficiency curve for pump {pump_id}")
        return 0

    def action_space(self):
        return gym.spaces.Discrete(len(self.action_map))

    def observation_space(self):
        num_state_elements = (
            len(self.pump_speeds)  # speeds
            + len(self.wds.junctions)  # pressures
            + len(self.wds.pumps)  # flows
            + len(self.wds.pumps))  # power

        return gym.spaces.Box(low=0.0, high=1.5, shape=(num_state_elements,), dtype=np.float32)


if __name__ == "__main__":
    
   
    env = wds(eff_weight=3.0, pressure_weight=1.0,demand_scale = 1.026884)
    # env = wds(eff_weight=3.0, pressure_weight=1)


    # # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(12)
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

    

    env = wds(eff_weight=3.0, pressure_weight=1.0,demand_scale = 1.026884)
    # env = wds(eff_weight=3.0, pressure_weight=1)


    env.step(8)
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

    # env = wds(eff_weight=3.0, pressure_weight=1.0)
    # env.step(40)
    # states = env.get_state()
    # reward = env._compute_reward()
    # print(f"Pump speeds: {env.pump_speeds}")
    # print()

    # print(f"Pump efficiencies: {env.pumpEffs}")
    # print()

    # print(f"Pump power: {env.pumpPower}")
    # print()

    # print(f"Valid heads ratio: {env.valid_heads_ratio}")
    # print(f"Eff ratio: {3*env.eff_ratio}")
    # print(f"Energy: {-0.02*env.total_power}")

    # print(f"Reward: {reward}")
    # print()

    # for i in range(len(env.action_map)):
    #     print(f"Action {i}: {env.action_map[i]}")