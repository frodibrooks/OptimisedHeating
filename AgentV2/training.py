import csv
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
from pump_env_demands import WdsWithDemand

# Adjusted speed levels for smaller action space
SPEED_LEVELS = np.round(np.arange(0.85, 1.25, 0.05), 3)
ACTION_MAP = [(s1, s2) for s1 in SPEED_LEVELS for s2 in SPEED_LEVELS]

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
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class Agent:
    def __init__(self, state_size, action_size, seed=0):
        random.seed(seed)
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100_000)
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.steps = 0
        self.action_size = action_size

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state)).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.steps += 1
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, torch.LongTensor(actions).unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    num_episodes = 2
    episode_len = 300  # Modify episode length here
    reward_log_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/training_results/reward_log_agent9.csv"
    demand_pattern = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/tests/demand_pattern_2024-11-03"

    with open(reward_log_path, mode='w', newline='') as file:
        csv.writer(file).writerow(['Episode', 'Total Reward'])

    # Initialize the WDS environment with demand pattern scaling and temporal variation
    env = WdsWithDemand(eff_weight=3.0, pressure_weight=1.0, episode_len=episode_len,demand_pattern=demand_pattern)

    state_size = len(env.reset())
    action_size = len(ACTION_MAP)
    agent = Agent(state_size, action_size)

    for episode in range(num_episodes):
        print()
        print(f"Episode {episode + 1}/{num_episodes}")
        print()
        state = env.reset()  # No demand pattern passed here
        total_reward = 0

        for step in range(episode_len):
            
            action_idx = agent.act(state)
            next_state, reward, done, _ = env.step(action_idx)
            agent.step(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            print(f"Step {step + 1}/{episode_len}")
            print(f"Pump speeds: {env.pump_speeds}")

            if done:
                break

        agent.decay_epsilon()

        with open(reward_log_path, mode='a', newline='') as file:
            csv.writer(file).writerow([episode + 1, total_reward])
        print(f"Episode {episode + 1}: Reward = {total_reward:.3f}, Epsilon = {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), "trained_model_vol9.pth")
    print("Model saved!")
