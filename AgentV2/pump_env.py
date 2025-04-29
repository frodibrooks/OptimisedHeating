import os
import numpy as np
import gym
from epynet import Network
from ecurves import eff_curves

class wds():
    def __init__(self,
                 wds_name="Vatnsendi",
                 speed_increment=0.05,
                 episode_len=300,
                 pump_groups=[['17', '10','25', '26'], ['27']],
                 total_demand_lo=0.8,
                 total_demand_hi=1.2,
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

        self.min_speed = 0.7
        self.max_speed = 1.4

        # New attributes for logging
        self.eff_ratio = 0.0
        self.valid_heads_ratio = 0.0

    def build_demand_dict(self):
        return {j.uid: j.basedemand for j in self.wds.junctions}

    def pump_power(self):
        self.pumpPower = [p.energy for p in self.wds.pumps.values()]

    def reset(self):
        self.pump_speeds = {pid: 1.0 for group in self.pumpGroups for pid in group}
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid]
        self.wds.solve()
        return self.get_state()

    def get_state(self):
        pump_speeds = list(self.pump_speeds.values())
        pressures = [j.pressure for j in self.wds.junctions]
        flows = [p.flow for p in self.wds.pumps.values()]
        return pump_speeds + pressures + flows

    def step(self, action_flat):
        group1 = action_flat % 3
        group2 = (action_flat // 3) % 3
        actions = [group1, group2]

        for i, group in enumerate(self.pumpGroups):
            group_action = actions[i]
            for pump_id in group:
                if group_action == 0:
                    self.pump_speeds[pump_id] -= self.speed_increment
                elif group_action == 2:
                    self.pump_speeds[pump_id] += self.speed_increment

                self.pump_speeds[pump_id] = np.round(self.pump_speeds[pump_id], 3)

        self.pump_speeds = {pid: np.clip(speed, self.min_speed, self.max_speed) for pid, speed in self.pump_speeds.items()}

        for pump_id, speed in self.pump_speeds.items():
            self.wds.links[pump_id].speed = speed

        demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

        self.wds.solve()

        self.calculate_pump_efficiencies()
        self.pump_power()

        pump_eff_ok = all(0 <= eff <= 1.2 for eff in self.pumpEffs.values())

        if pump_eff_ok:
            heads = np.array([j.pressure for j in self.wds.junctions])
            self.valid_heads_ratio = np.mean(heads >= self.headLimitLo)

            total_efficiency = np.prod(list(self.pumpEffs.values()))
            self.eff_ratio = np.clip(total_efficiency / self.peakTotEff, 0, 1)

            reward = (self.eff_weight * self.eff_ratio) + (self.pressure_weight * self.valid_heads_ratio)
        else:
            reward = 0.0
            self.eff_ratio = 0.0
            self.valid_heads_ratio = 0.0

        done = False
        return self.get_state(), reward, done, {}

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
                efficiency = float(curve(flow_lps))
                return efficiency
            except Exception as e:
                print(f"Error calculating efficiency for pump {pump_id}, flow={flow_lps}: {e}")
                return 0
        else:
            raise ValueError(f"No efficiency curve for pump {pump_id}")

    def action_space(self):
        return gym.spaces.Discrete(9)

    def observation_space(self):
        num_state_elements = len(self.pump_speeds) + len(self.wds.junctions) + len(self.wds.pumps)
        return gym.spaces.Box(low=0.0, high=1.5, shape=(num_state_elements,), dtype=np.float32)
