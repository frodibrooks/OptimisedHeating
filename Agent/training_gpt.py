import os
import numpy as np
import scipy.stats as stats
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from epynet import Network
import gym.spaces
from opti_algorithms import nm
from scipy.optimize import minimize

class wds():
    def __init__(self,
                 wds_name="Vatnsendi_dummy_ecurves",
                 speed_increment=0.05,
                 episode_len=10,
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
        # Create the demand dictionary
        demandDict = {}
        for junction in self.wds.junctions:
            demandDict[junction.uid] = junction.basedemand
        return demandDict

    def reset(self):
        self.pump_speeds = np.ones(shape=(len(self.pumpGroups)), dtype=np.float32)
        # Reset demands as well if needed
        return self.pump_speeds

    def step(self, action):
        # Update pump speeds based on the action
        self.pump_speeds += action * self.speed_increment

        # Ensure pump speeds stay within a feasible range
        self.pump_speeds = np.clip(self.pump_speeds, 0.1, 2.0)

        # Update demands (based on random fluctuation for the example)
        demand_scale = np.random.uniform(self.total_demand_lo, self.total_demand_hi)
        self.demandDict = {k: v * demand_scale for k, v in self.demandDict.items()}

        # Simulate network (the system state would be evaluated here)
        reward = -np.sum(self.pump_speeds)  # Reward is negative pump usage for minimization

        done = False  # Can include condition to terminate episode
        return np.array(self.pump_speeds), reward, done, {}

    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(len(self.pumpGroups),), dtype=np.float32)

    def observation_space(self):
        return gym.spaces.Box(low=0, high=2, shape=(len(self.pumpGroups),), dtype=np.float32)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Initialize policy and target networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0  # epsilon-greedy exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.uniform(-1, 1, size=self.action_size)  # Random action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).numpy()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q values for current states
        q_values = self.policy_net(states).gather(1, actions.long().unsqueeze(-1))
        
        # Compute target Q values (using the target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize environment and agent
env = wds()
agent = Agent(state_size=len(env.pumpGroups), action_size=len(env.pumpGroups))

# Training loop
num_episodes = 500
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
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
