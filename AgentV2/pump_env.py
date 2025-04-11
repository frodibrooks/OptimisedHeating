from epynet import Network
import numpy as np
import gym
import gym.spaces
import os
from ecurves import eff_curves  # Ensure this contains proper polynomial functions


class wds():
    def __init__(self,
                 wds_name="Vatnsendi_dummy_ecurves",
                 speed_increment=0.05,
                 episode_len=20,
                 pump_groups=[['17', '10', '25', '26', '27']],
                 total_demand_lo=0.8,
                 total_demand_hi=1.2,
                 reset_orig_pump_speeds=False,
                 reset_orig_demands=False,
                 seed=None,
                 eff_weight=1.0,
                 pressure_weight=1.0):

        if seed:
            np.random.seed(seed)

        pathToRoot = os.path.dirname(os.path.realpath(__file__))
        pathToWDS = os.path.join(pathToRoot, "water_network", wds_name + ".inp")
        self.wds = Network(pathToWDS)

        self.demandDict = self.build_demand_dict()
        self.pumpGroups = pump_groups
        self.pump_speeds = np.ones(shape=(len(self.pumpGroups)), dtype=np.float32)
        self.pumpEffs = np.ones_like(self.pump_speeds)

        self.episode_len = episode_len
        self.total_demand_lo = total_demand_lo
        self.total_demand_hi = total_demand_hi
        self.speed_increment = speed_increment

        # Reward config
        self.headLimitLo = 35  # Minimum pressure (in meters)
        self.peakTotEff = 0.0629
        self.eff_weight = eff_weight
        self.pressure_weight = pressure_weight

    def build_demand_dict(self):
        return {j.uid: j.basedemand for j in self.wds.junctions}

    def reset(self):
        self.pump_speeds[:] = 1.0
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid]
        self.wds.solve()
        return self.pump_speeds.copy()

    def step(self, action):
        # Update pump speeds based on action
        if action == 0:
            self.pump_speeds *= (1 - self.speed_increment)
        elif action == 1:
            pass  # No change
        elif action == 2:
            self.pump_speeds *= (1 + self.speed_increment)

        self.pump_speeds = np.clip(self.pump_speeds, 0.8, 1.3)

        # Apply pump speeds to model
        for group, speed in zip(self.pumpGroups, self.pump_speeds):
            for pid in group:
                self.wds.links[pid].speed = speed

        # Randomize demands
        demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        # Solve network
        self.wds.solve()

        # Calculate efficiencies
        self.calculate_pump_efficencies()
        pump_eff_ok = (np.array(self.pumpEffs) >= 0).all() and (np.array(self.pumpEffs) <= 1.2).all()


        if pump_eff_ok:
            heads = np.array([j.head for j in self.wds.junctions])
            valid_heads_count = np.sum(heads >= self.headLimitLo)
            total_heads_count = len(heads)
            valid_heads_ratio = valid_heads_count / total_heads_count
           


            # Normalized reward components
            total_efficiency = np.prod(self.pumpEffs)
            eff_ratio = np.clip(total_efficiency / self.peakTotEff, 0, 1)
            pressure_score = np.clip(valid_heads_ratio, 0, 1)

            # Weighted reward
            reward = (
                self.eff_weight * eff_ratio +
                self.pressure_weight * pressure_score
            )

            # Penalty for using higher pump speeds
            reward -= np.sum(self.pump_speeds) * 0.1
        else:
            reward = 0
            valid_heads_ratio = 0

        done = False
        return self.pump_speeds.copy(), reward, done, {}

    def calculate_pump_efficencies(self):
        self.pumpEffs = []  # Initialize the list to store individual efficiencies
        for group in self.pumpGroups:
            for pid in group:
                pump = self.wds.pumps[pid]
                try:
                    flow_lps = pump.flow  # Already in LPS
                    eff = self.calculate_efficiency(pid, flow_lps)
                    self.pumpEffs.append(eff)  # Append individual efficiency
                except Exception as e:
                    print(f"Error getting efficiency for pump {pid}: {e}")
                    self.pumpEffs.append(0)  # Append 0 if there's an error


    def calculate_efficiency(self, pump_id, flow_lps):
        curve = eff_curves.get(pump_id)
        if curve:
            try:
                efficiency = float(curve(flow_lps))
                return efficiency
            except Exception as e:
                print(f"Error calculating efficiency for pump {pump_id}, Flow: {flow_lps}: {e}")
                return 0
        else:
            raise ValueError(f"No efficiency curve for pump {pump_id}")

    def action_space(self):
        return gym.spaces.Discrete(3)

    def observation_space(self):
        return gym.spaces.Box(low=0.8, high=1.3, shape=(len(self.pumpGroups),), dtype=np.float32)


# Debugging prints added below
env = wds()
env.reset()  # <--- This is important
obs, reward, done, info = env.step(1)
 # Steady state, no change in pump speeds
print(f"Reward: {reward}")

# Additional debugging prints to track variables
print("------ Debugging Output ------")

# # Printing demand values before and after scaling
# demand_scale = np.random.uniform(env.total_demand_lo, env.total_demand_hi)
# print(f"Demand scale: {demand_scale}")
# for junction in env.wds.junctions:
#     print(f"Junction {junction.uid} original demand: {env.demandDict[junction.uid]}, scaled demand: {junction.basedemand}")

# Printing the efficiency values
print(f"Pump efficiencies: {env.pumpEffs}")

# Printing the head pressures
heads = np.array([j.head for j in env.wds.junctions])
print(f"Heads at junctions: {heads}")

# Printing the valid heads ratio
valid_heads_ratio = np.mean(heads >= env.headLimitLo)
print(f"Valid heads ratio: {valid_heads_ratio}")

# Printing reward components
total_efficiency = np.prod(env.pumpEffs)
eff_ratio = np.clip(total_efficiency / env.peakTotEff, 0, 1)

print(f"Pressure score is {total_efficiency} divided by {env.peakTotEff} = {eff_ratio}")