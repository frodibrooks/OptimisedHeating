import csv
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
from pump_env import wds  # Custom environment

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_fc = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
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
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.steps = 0
        self.action_size = action_size

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Return flat index

        with torch.no_grad():
            action_values = self.policy_net(state)
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.steps += 1
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).view(self.batch_size, -1)
        next_states = torch.FloatTensor(np.array(next_states)).view(self.batch_size, -1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)

        q_values = q_values.gather(1, actions_tensor)
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets.view(-1, 1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# === Training Process ===
env = wds(eff_weight=3.0, pressure_weight=1.0)
initial_state = env.reset()
state_size = len(initial_state)
action_size = 2 * 3  # 3 groups × 3 actions = 9

agent = Agent(state_size=state_size, action_size=action_size)

reward_file_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\training_results\reward_log.csv"
with open(reward_file_path, mode='w', newline='') as file:
    csv.writer(file).writerow(['Episode', 'Total Reward'])

num_episodes = 200
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(env.episode_len):
        flat_action = agent.act(state)
        # Convert flat action index to multi-action for the environment
        group1 = flat_action % 3
        group2 = (flat_action // 3) % 3
        multi_action = [group1, group2]

        next_state, reward, done, _ = env.step(multi_action)
        agent.step(state, flat_action, reward, next_state, done)
        state = next_state
        total_reward += reward

    with open(reward_file_path, mode='a', newline='') as file:
        csv.writer(file).writerow([episode + 1, total_reward])

    print(f"Episode {episode + 1}: Total Reward = {total_reward:.3f}")






torch.save(agent.policy_net.state_dict(), "trained_model.pth")
print("Model saved!")

