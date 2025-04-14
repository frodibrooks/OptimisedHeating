import os
import numpy as np
import gym
from epynet import Network
from ecurves import eff_curves  # Ensure this contains proper polynomial functions

class wds():
    def __init__(self,
                 wds_name="Vatnsendi_dummy_ecurves",
                 speed_increment=0.05,
                 episode_len=200,
                 pump_groups=[['17', '10'], ['25', '26'], ['27']],
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

        # Store individual pump speeds and efficiencies
        self.pump_speeds = {pid: 1.0 for group in pump_groups for pid in group}
        self.pumpEffs = {pid: 1.0 for group in pump_groups for pid in group}

        self.episode_len = episode_len
        self.total_demand_lo = total_demand_lo
        self.total_demand_hi = total_demand_hi
        self.speed_increment = speed_increment

        # Reward config
        self.headLimitLo = 35  # Minimum pressure (in meters)
        self.peakTotEff = 0.0629
        self.eff_weight = eff_weight
        self.pressure_weight = pressure_weight

        # Define speed limits
        self.min_speed = 0.7
        self.max_speed = 1.4

    def build_demand_dict(self):
        return {j.uid: j.basedemand for j in self.wds.junctions}

    def reset(self):
        self.pump_speeds = {pid: 1.0 for group in self.pumpGroups for pid in group}
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid]
        self.wds.solve()

        return self.get_state()

    def get_state(self):
        pump_speeds = list(self.pump_speeds.values())
        pressures = [j.head for j in self.wds.junctions]
        flows = [p.flow for p in self.wds.pumps.values()]
        return pump_speeds + pressures + flows

    def step(self, action):
        # Ensure valid shape and clip action values
        action = np.clip(action, 0, 2)

        # Each action[i] applies to group i
        for i, group in enumerate(self.pumpGroups):
            group_action = action[i]
            for pump_id in group:
                # Adjusting pump speeds by fixed value (0.05)
                if group_action == 0:
                    self.pump_speeds[pump_id] -= self.speed_increment  # Decrease speed by fixed amount
                elif group_action == 1:
                    pass  # No change
                elif group_action == 2:
                    self.pump_speeds[pump_id] += self.speed_increment  # Increase speed by fixed amount

        # Clamp pump speeds to the defined range [0.85, 1.3]
        self.pump_speeds = {pid: np.clip(speed, self.min_speed, self.max_speed) for pid, speed in self.pump_speeds.items()}

        # Apply to model
        for pump_id, speed in self.pump_speeds.items():
            self.wds.links[pump_id].speed = speed

        # Randomize demand
        demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        # Solve network
        self.wds.solve()

        # Calculate efficiencies
        self.calculate_pump_efficencies()

        # Compute reward
       # Compute reward
        pump_eff_ok = all(0 <= eff <= 1.2 for eff in self.pumpEffs.values())

        if pump_eff_ok:
            heads = np.array([j.head for j in self.wds.junctions])
            valid_heads_ratio = np.mean(heads >= self.headLimitLo)

            total_efficiency = np.prod(list(self.pumpEffs.values()))
            eff_ratio = np.clip(total_efficiency / self.peakTotEff, 0, 1)

            if valid_heads_ratio < 1.0:
                # Big penalty if any node has pressure below threshold
                reward = -10.0
            else:
                # Pressure constraint met, reward with scaled efficiency
                reward = eff_ratio * 100.0  # scale for better gradient
                print(f"Efficiency ratio: {eff_ratio:.4f}  Pressure ratio {valid_heads_ratio:.4f}")
        else:
            reward = 0.0


        done = False
        return self.get_state(), reward, done, {}

    def calculate_pump_efficencies(self):
        self.pumpEffs = {}
        for group in self.pumpGroups:
            for pump_id in group:
                pump = self.wds.pumps[pump_id]
                try:
                    flow_lps = pump.flow
                    eff = self.calculate_efficiency(pump_id, flow_lps)
                    self.pumpEffs[pump_id] = eff
                except Exception as e:
                    print(f"Error getting efficiency for pump {pump_id}: {e}")
                    self.pumpEffs[pump_id] = 0

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
        return gym.spaces.MultiDiscrete([3] * len(self.pumpGroups))

    def observation_space(self):
        num_state_elements = len(self.pump_speeds) + len(self.wds.junctions) + len(self.wds.pumps)
        return gym.spaces.Box(low=0.0, high=1.3, shape=(num_state_elements,), dtype=np.float32)


# # Debugging prints added below
# env = wds()
# env.reset()  # <--- This is important
# obs, reward, done, info = env.step([1,1,2])  # Example action for each pump (no change, decrease, increase)
# # Steady state, no change in pump speeds
# print(f"Reward: {reward}")

# # Additional debugging prints to track variables
# print("------ Debugging Output ------")

# # Printing the efficiency values
# print(f"Pump efficiencies: {env.pumpEffs}")

# # Printing the head pressures
# heads = np.array([j.head for j in env.wds.junctions])
# print(f"Heads at junctions: {heads}")

# # Printing the valid heads ratio
# valid_heads_ratio = np.mean(heads >= env.headLimitLo)
# print(f"Valid heads ratio: {valid_heads_ratio}")

# # Printing reward components
# total_efficiency = np.prod(list(env.pumpEffs.values()))
# eff_ratio = np.clip(total_efficiency / env.peakTotEff, 0, 1)
# print(f"Pressure score is {total_efficiency} divided by {env.peakTotEff} = {eff_ratio}")

# # Printing pump IDs and their respective speeds
# print("------ Pump ID and Speeds ------")
# for pump_id, speed in env.pump_speeds.items():
#     print(f"Pump ID: {pump_id}, Speed: {speed:.3f}")

# state = env.get_state()
# print(state[:5])
