import pandas as pd
import numpy as np
from pump_env import wds

class WdsWithDemand(wds):
    def __init__(self, action_map=None, demand_pattern=None, eff_weight=3.0, pressure_weight=1.0, episode_len=300, *args, **kwargs):
        # Initialize the parent class, passing only the necessary arguments
        super().__init__(episode_len=episode_len, eff_weight=eff_weight, pressure_weight=pressure_weight, *args, **kwargs)

        # If action_map is passed, use it; otherwise, create the default action_map
        self.action_map = action_map if action_map is not None else [
            (i, j)
            for i in range(len(np.round(np.arange(0.8, 1.3001, 0.025), 3)))
            for j in range(len(np.round(np.arange(0.8, 1.3001, 0.025), 3)))
        ]

        # Define speed levels for the pumps
        self.speed_levels = np.round(np.arange(0.8, 1.3001, 0.025), 3)
        
        # Debugging print to check the ACTION_MAP and SPEED_LEVELS
        # print(f"Action Map: {self.action_map}")
        # print(f"Speed Levels: {self.speed_levels}")
        
        # Load the demand pattern if provided
        self.demand_pattern = self._load_pattern(demand_pattern)
        self.demand_index = 0

    def _load_pattern(self, pattern):
        """Load a demand pattern from a CSV or use the provided numpy array."""
        if isinstance(pattern, str):  # Assume CSV path
            return pd.read_csv(pattern)['demand_pattern'].values
        elif isinstance(pattern, np.ndarray):  # Already an array
            return pattern
        else:
            return None

    def reset(self, demand_pattern=None):
        """Reset the environment with the given or current demand pattern."""
        self.demand_index = 0
        if demand_pattern is not None:
            self.demand_pattern = self._load_pattern(demand_pattern)
        return super().reset()

    def action_index_to_list(self, action_idx):
        """Convert action index to a tuple of pump speed indices."""
        # Debugging print statement to show the action_idx and the mapped action
        # print(f"Converting action_idx {action_idx} to speed levels")
        return self.action_map[action_idx]

    def step(self, action_idx):
        """Take a step in the environment based on the selected action."""
        # Debugging print statements
        # print(f"Action Index: {action_idx}")  # Print the action index received
        idx1, idx2 = self.action_index_to_list(action_idx)
        # print(f"Action Index Mapped: idx1 = {idx1}, idx2 = {idx2}")  # Print idx1 and idx2 values

        # Ensure that idx1 and idx2 are integers
        # print(f"Speed Levels: {self.speed_levels}")
        
        # Convert speed values to their indices in the speed_levels array
        try:
            # Ensure the values are valid and match the speed_levels array
            idx1 = np.where(self.speed_levels == idx1)[0][0]
            idx2 = np.where(self.speed_levels == idx2)[0][0]
            # print(f"Converted idx1 = {idx1}, idx2 = {idx2}")
            speed1 = self.speed_levels[idx1]
            speed2 = self.speed_levels[idx2]
            # print(f"Selected Speeds: speed1 = {speed1}, speed2 = {speed2}")
        except IndexError as e:
            print(f"Error: {e}, idx1 = {idx1}, idx2 = {idx2}, speed_levels length = {len(self.speed_levels)}")

        # Apply the chosen speeds to the pumps
        for group_idx, speed in zip(range(len(self.pumpGroups)), [speed1, speed2]):
            for pump_id in self.pumpGroups[group_idx]:
                self.pump_speeds[pump_id] = speed
                self.wds.pumps[pump_id].speed = speed


        # Set the demand scale for the current timestep
        if self.demand_pattern is not None and self.demand_index < len(self.demand_pattern):
            demand_scale = self.demand_pattern[self.demand_index]
            self.demand_index += 1
        else:
            # If no pattern or if we run out, use a random demand scale
            demand_scale = 1

        # Adjust the demand for all junctions
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid] * demand_scale

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

if __name__ == "__main__":
        
    SPEED_LEVELS = np.round(np.arange(0.8, 1.301, 0.025), 3)
    ACTION_MAP = [(s1, s2) for s1 in SPEED_LEVELS for s2 in SPEED_LEVELS]
   
    env = WdsWithDemand(action_map=ACTION_MAP, eff_weight=3.0, pressure_weight=1.0, episode_len=300)

    # Gott dæmi um að ecurves gefa betra reward en nsamt er consumed power meira 

    env.step(168)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Pump efficiencies: {env.pumpEffs}")
    print()

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Reward: {reward}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Eff ratio: {3*env.eff_ratio}")


    print(f"States: {states[:10]}")

    env.step(176)
    states = env.get_state()
    reward = env._compute_reward()
    print(f"Pump speeds: {env.pump_speeds}")
    print()

    print(f"Pump efficiencies: {env.pumpEffs}")
    print()

    print(f"Pump power: {env.pumpPower}")
    print()

    print(f"Reward: {reward}")
    print()

    print(f"Valid heads ratio: {env.valid_heads_ratio}")
    print(f"Eff ratio: {3*env.eff_ratio}")


    print(f"States: {states[:10]}")
    # for i in range(len(ACTION_MAP)):
    #     print(f"Action {i}: {ACTION_MAP[i]}")
