import gym
from gym import spaces  # Import spaces from gym
from epyt import epanet
import numpy as np


def compute_reward(network, min_pressure=3.5):
    power_penalty = sum(network.pumps[p].power for p in network.pumps.keys())  # Penalize power usage
    pressure_penalty = sum(max(0, min_pressure - network.nodes[n].pressure) for n in network.nodes.keys())

    return -power_penalty - (pressure_penalty * 10)  # 10x weight on pressure violations


class PumpEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.network = epanet.Network("network.inp")  # Load EPANET model
        self.pump_ids = list(self.network.pumps.keys())
        self.node_ids = list(self.network.nodes.keys())

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.node_ids) + len(self.pump_ids) * 3,))
        self.action_space = spaces.MultiDiscrete([3] * len(self.pump_ids))  # 3 actions per pump

    def step(self, action):
        for i, pump in enumerate(self.pump_ids):
            speed_change = [-0.05, 0, 0.05][action[i]]
            self.network.pumps[pump].speed *= (1 + speed_change)
        
        self.network.solve()  # Run EPANET simulation

        state = self._get_state()
        reward = compute_reward(self.network)  # Calling the function defined before the class
        done = False

        return state, reward, done, {}

    def reset(self):
        self.network.reset()  # Reset network to initial state
        return self._get_state()

    def _get_state(self):
        pressures = [self.network.nodes[n].pressure for n in self.node_ids]
        speeds = [self.network.pumps[p].speed for p in self.pump_ids]
        flows = [self.network.pumps[p].flow for p in self.pump_ids]
        power = [self.network.pumps[p].power for p in self.pump_ids]
        return np.array(pressures + speeds + flows + power, dtype=np.float32)
