import csv
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
from pump_env import wds  # Assuming this is the custom environment you created

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream
        self.value_fc = nn.Linear(128, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        value = self.value_fc(x)  # Shape: (batch_size, 1)
        advantage = self.advantage_fc(x)  # Shape: (batch_size, action_size)

        # Combine value and advantage into Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class Agent:
    def __init__(self, state_size, action_size, seed=0):
        random.seed(seed)
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100_000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.steps = 0

    def act(self, state):
        # Convert state to tensor if it's a list (handle case when state is a list)
        if isinstance(state, list):
            state = torch.FloatTensor(state)

        if random.random() < self.epsilon:
            # Return a random action for each of the 3 pump groups (each group has 3 actions)
            return [random.choice([0, 1, 2]) for _ in range(3)]  # 3 actions for 3 groups
        
        state = state.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action_values = self.policy_net(state)
            # Select action for each of the 3 pump groups
            return [action_values[0, i:i+3].argmax().item() for i in range(0, action_values.shape[1], 3)]  # 3 actions for 3 groups

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.steps += 1
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))  # Shape: (batch_size, state_size)
        actions = torch.LongTensor(actions)  # Shape will be (batch_size, num_pump_groups)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        states = states.view(self.batch_size, -1)  # Flatten states if necessary

        # Get the Q values for the selected actions
        q_values = self.policy_net(states)  # Shape: (batch_size, action_size)

        # Flatten q_values and actions for gathering
        q_values = q_values.gather(1, actions.view(-1, 1))  # Select Q-values for the chosen actions
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        # Calculate the target values
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and backpropagate
        loss = nn.MSELoss()(q_values, targets.view(-1, 1))  # Ensure targets have shape (batch_size, 1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Update epsilon for exploration-exploitation balance
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# === Training Process ===
env = wds(eff_weight=1.0, pressure_weight=1.0)

# Number of pump groups (3 groups of pumps)
num_pump_groups = 3
actions_per_group = 3  # 3 possible actions per group (decrease, keep, increase)
action_size = num_pump_groups * actions_per_group  # This will be 9

agent = Agent(state_size=len(env.pumpGroups), action_size=action_size)

reward_file_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\training_results\reward_log.csv"
with open(reward_file_path, mode='w', newline='') as file:
    csv.writer(file).writerow(['Episode', 'Total Reward'])

num_episodes = 20
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(env.episode_len):
        action = agent.act(state)  # Action for 3 pump groups

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Print the pump group speeds after each action
        pump_speeds = state[:5]  # Assuming the first part of the state contains pump speeds
        print(f"Episode {episode+1}, Step {t+1}: Pump Group Speeds: {pump_speeds}")

    # Log the total reward for the episode
    with open(reward_file_path, mode='a', newline='') as file:
        csv.writer(file).writerow([episode, total_reward])

    print(f"Episode {episode+1}: Total Reward = {total_reward:.3f}")
