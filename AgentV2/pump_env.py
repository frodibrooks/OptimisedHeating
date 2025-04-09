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
                 seed=None):

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

        # Constants for reward
        self.headLimitLo = 35  # Bar, example minimum head pressure
        self.peakTotEff = 0.85 ** len(self.pumpGroups)  # Best case efficiency product

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
        pump_eff_ok = (self.pumpEffs >= 0).all() and (self.pumpEffs <= 1.2).all()

        if pump_eff_ok:
            heads = np.array([j.head for j in self.wds.junctions])
            valid_heads_ratio = np.mean(heads >= self.headLimitLo)

            # Base reward from efficiency
            total_efficiency = np.prod(self.pumpEffs)
            eff_ratio = total_efficiency / self.peakTotEff

            reward = eff_ratio

            # Pressure quality reward/penalty
            if valid_heads_ratio == 1.0:
                reward += 5
            elif valid_heads_ratio < 0.8:
                reward -= 3
            else:
                reward += valid_heads_ratio  # Mild reward between 0 and 1

            # Penalty for total pump usage
            reward -= np.sum(self.pump_speeds) * 0.1

            print(f"Efficiency ratio: {eff_ratio:.3f}, Valid heads ratio: {valid_heads_ratio:.3f}, Reward: {reward:.3f}", flush=True)
        else:
            reward = 0
            valid_heads_ratio = 0

        done = False
        return self.pump_speeds.copy(), reward, done, {}

    def calculate_pump_efficencies(self):
        self.pumpEffs = []
        for group, speed in zip(self.pumpGroups, self.pump_speeds):
            effs = []
            for pid in group:
                eff = self.calculate_efficiency(pid, speed)
                effs.append(eff)
            self.pumpEffs.append(np.mean(effs))
        self.pumpEffs = np.array(self.pumpEffs)

    def calculate_efficiency(self, pump_id, speed):
        curve = eff_curves.get(pump_id)
        if curve:
            return float(curve(speed))
        else:
            raise ValueError(f"No efficiency curve for pump {pump_id}")

    def action_space(self):
        return gym.spaces.Discrete(3)

    def observation_space(self):
        return gym.spaces.Box(low=0.8, high=1.3, shape=(len(self.pumpGroups),), dtype=np.float32)

# Debugging prints added below
env = wds()
obs, reward, done, info = env.step(1)  # Steady state, no change in pump speeds
print(f"Reward: {reward}")

# Additional debugging prints to track variables
print("------ Debugging Output ------")

# Printing demand values before and after scaling
demand_scale = np.random.uniform(env.total_demand_lo, env.total_demand_hi)
print(f"Demand scale: {demand_scale}")
for junction in env.wds.junctions:
    print(f"Junction {junction.uid} original demand: {env.demandDict[junction.uid]}, scaled demand: {junction.basedemand}")

# Printing the efficiency values
print(f"Pump efficiencies: {env.pumpEffs}")

# Printing the head pressures
heads = np.array([j.head for j in env.wds.junctions])
print(f"Heads at junctions: {heads}")

# Printing the valid heads ratio
valid_heads_ratio = np.mean(heads >= env.headLimitLo)
print(f"Valid heads ratio: {valid_heads_ratio}")
