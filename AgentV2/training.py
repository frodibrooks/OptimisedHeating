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
        if random.random() < self.epsilon:
            # Return a random action for each of the 5 pumps
            return [random.choice([0, 1, 2]) for _ in range(5)]  
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state)
            # Select action for each of the 5 pumps
            return [action_values[0, i:i+3].argmax().item() for i in range(0, action_values.shape[1], 3)]

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
        actions = torch.LongTensor(actions)  # Shape will be (batch_size, num_pumps)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Reshape the states to ensure batch_size x state_size
        states = states.view(self.batch_size, -1)  # Flatten states if necessary

        # Reshape actions to match the q_values shape
        actions = actions.view(self.batch_size, -1)  # Shape should be (batch_size, num_pumps)

        # Get the Q values for the selected actions
        q_values = self.policy_net(states)  # Shape: (batch_size, action_size)
        
        # Flatten q_values and actions for gathering
        q_values = q_values.gather(1, actions)  # Select Q-values for the chosen actions
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


# === Train Agent ===
env = wds(eff_weight=2.0, pressure_weight=1.0)

# Here we define action size as 5 pumps * 3 actions each = 15 actions total
num_pumps = 5
actions_per_pump = 3
action_size = num_pumps * actions_per_pump  # This will be 15

agent = Agent(state_size=len(env.pumpGroups), action_size=action_size)

reward_file_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\training_results\reward_log.csv"
with open(reward_file_path, mode='w', newline='') as file:
    csv.writer(file).writerow(['Episode', 'Total Reward'])

num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    episode_actions = []  # List to store the actions for this episode
    for t in range(env.episode_len):
        # Make action an array of actions for each of the 5 pumps
        action = agent.act(state)

        # # Printing action details
        # action_details = []
        # for pump_idx, pump_action in enumerate(action):
        #     if pump_action == 0:
        #         action_details.append(f"Pump {pump_idx+1}: Decrease speed")
        #     elif pump_action == 1:
        #         action_details.append(f"Pump {pump_idx+1}: Keep speed")
        #     elif pump_action == 2:
        #         action_details.append(f"Pump {pump_idx+1}: Increase speed")
        # print(f"Episode {episode+1}, Step {t+1}: {', '.join(action_details)}")

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        # episode_actions.append(action_details)

    # Log the total reward for the episode
    with open(reward_file_path, mode='a', newline='') as file:
        csv.writer(file).writerow([episode, total_reward])
    
    print(f"Episode {episode+1}: Total Reward = {total_reward:.3f}")
