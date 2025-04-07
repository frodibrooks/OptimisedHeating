import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
from epynet import Network
import gym.spaces
import csv

class wds():
    def __init__(self,
                 wds_name="Vatnsendi_dummy_ecurves",
                 speed_increment=0.05,
                 episode_len=20,  # Increased episode length for more training time
                 pump_groups=[['17', '10', '25', '26', '27']],
                 total_demand_lo=0.7,
                 total_demand_hi=1.3,
                 reset_orig_pump_speeds=False,
                 reset_orig_demands=False,
                 seed=None):
        self.seedNum = seed
        if self.seedNum:
            np.random.seed(self.seedNum)
        else:
            np.random.seed()

        pathToRoot = os.path.dirname(os.path.realpath(__file__))
        pathToWDS = os.path.join(pathToRoot, "water_network", wds_name + ".inp")

        self.wds = Network(pathToWDS)
        self.demandDict = self.build_demand_dict()
        self.pumpGroups = pump_groups
        self.pump_speeds = np.ones(shape=(len(self.pumpGroups)), dtype=np.float32)
        self.pumpEffs = np.empty(shape=(len(self.pumpGroups)), dtype=np.float32)

        self.episode_len = episode_len
        self.total_demand_lo = total_demand_lo
        self.total_demand_hi = total_demand_hi
        self.speed_increment = speed_increment

    def build_demand_dict(self):
        demandDict = {}
        for junction in self.wds.junctions:
            demandDict[junction.uid] = junction.basedemand
        return demandDict

    def reset(self):
        self.pump_speeds = np.ones(shape=(len(self.pumpGroups)), dtype=np.float32)
        return self.pump_speeds

    def step(self, action):
        # Adjust pump speeds based on the action taken
        if action == 0:
            self.pump_speeds *= (1 - 0.005)  # Decrease pump speeds
        elif action == 1:
            self.pump_speeds *= 1  # Keep pump speeds the same
        elif action == 2:
            self.pump_speeds *= (1 + 0.005)  # Increase pump speeds

        # Clip pump speeds to be within the valid range (0.8 to 1.3)
        self.pump_speeds = np.clip(self.pump_speeds, 0.8, 1.3)

        # Randomly scale the demands in the system
        demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)
        self.demandDict = {k: v * demand_scale for k, v in self.demandDict.items()}

        # Calculate pump efficiencies (using their efficiency curves)
        self.calculate_pump_efficencies()

        # Check pump efficiency: all pumps should be within the efficiency range [0, 1]
        pump_eff_ok = (self.pumpEffs < 1).all() and (self.pumpEffs > 0).all()

        if pump_eff_ok:
            # Calculate head values for all junctions
            heads = np.array([junction.head for junction in self.wds.junctions])
            
            # Count the number of junctions with pressure below the threshold
            invalid_heads_count = np.count_nonzero(heads < self.headLimitLo)
            valid_heads_count = len(heads) - invalid_heads_count

            # Calculate the ratio of valid junctions (pressures within the acceptable range)
            valid_heads_ratio = valid_heads_count / len(heads)

            # Calculate the total demand in the system
            total_demand = sum(junction.basedemand for junction in self.wds.junctions)

            # Calculate the overall pump efficiency (product of individual efficiencies)
            total_efficiency = np.prod(self.pumpEffs)

            # Efficiency ratio: comparison with the peak efficiency
            eff_ratio = total_efficiency / self.peakTotEff

            # Reward function incorporates both efficiency and pressure satisfaction
            reward = (eff_ratio * valid_heads_ratio) - np.sum(self.pump_speeds) * 0.1

        else:
            # If pumps are not operating within valid efficiency bounds, the reward is 0
            reward = 0
            valid_heads_ratio = 0

        done = False  # Episode doesn't end until max length is reached
        return np.array(self.pump_speeds), reward, done, {}

    def calculate_pump_efficencies(self):
        # Assuming we have pump efficiency curves or a similar function to calculate efficiency
        # This function should calculate the efficiencies of the pumps based on their current speeds
        self.pumpEffs = np.array([self.calculate_efficiency(speed) for speed in self.pump_speeds])

    def calculate_efficiency(self, speed):
        # This is a placeholder for the actual efficiency curve based on pump speed
        # A simple linear relationship could be used, but it should be replaced with actual data
        return 1 - 0.1 * (speed - 1)**2  # Example efficiency curve (quadratic drop with speed)


    def action_space(self):
        return gym.spaces.Discrete(3)

    def observation_space(self):
        return gym.spaces.Box(low=0.8, high=1.3, shape=(len(self.pumpGroups),), dtype=np.float32)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Increased layer size for better capacity
        self.fc2 = nn.Linear(128, 128)        # Increased layer size
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)  # Lower learning rate for finer updates
        self.memory = deque(maxlen=100000)  # Larger memory buffer for more experience
        self.batch_size = 128  # Larger batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])  # Random action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = wds()
agent = Agent(state_size=len(env.pumpGroups), action_size=env.action_space().n)

reward_file_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\training_results\reward_log.csv"

with open(reward_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Total Reward'])

num_episodes = 1000  # Increased number of episodes
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(env.episode_len):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    with open(reward_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode + 1, total_reward])

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
