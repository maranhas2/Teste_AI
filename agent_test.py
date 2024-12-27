import gymnasium as gym
import rsoccer_gym
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Using VSS Single Agent env
env = gym.make('VSS-v0', render_mode="rgb_array")

# Neural network model for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 100
tempo = time.time()

# Initialize Q-networks
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

def select_action(state, epsilon):
    if random.random() < epsilon:
        # Explore: sample random action from the Box action space
        return env.action_space.sample()
    else:
        # Exploit: use the policy network to predict the action
        state = torch.FloatTensor(state).unsqueeze(0)
        action = policy_net(state).detach().squeeze(0).numpy()
        return np.clip(action, env.action_space.low, env.action_space.high)  # Clip to valid action range


def optimize_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.FloatTensor(np.array(action_batch))
    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch))
    done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

    # Compute predicted Q-values (actions) for the current states
    predicted_actions = policy_net(state_batch)

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_actions = target_net(next_state_batch)
        target_q_values = reward_batch + gamma * (1 - done_batch) * next_actions.max(dim=1, keepdim=True)[0]

    # Compute loss
    loss = nn.MSELoss()(predicted_actions, action_batch)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main training loop
rewards_per_episode = []
steps_done = 0

for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done and not (time.time() - tempo) > 20:

        # Select action
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        
        # Store transition in memory
        memory.append((state, action, reward, next_state, done))
        
        # Update state
        state = next_state
        episode_reward += reward
        
        # Optimize model
        optimize_model()

        # Update target network periodically
        if steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    print(episode_reward, (time.time() - tempo))
    tempo = time.time()
    rewards_per_episode.append(episode_reward)

# Plotting the rewards per episode
import matplotlib.pyplot as plt
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN on VSS Single Agent')
plt.show()

def test_agent(num_episodes=10):
    """
    Tests the trained agent on the environment for a specified number of episodes.
    """
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # Select action using the policy network (no epsilon exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy_net(state_tensor).squeeze(0).numpy()
            
            # Clip the action to ensure it's within the valid range
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take a step in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            
            # Update state and accumulate rewards
            state = next_state
            episode_reward += reward

            # Stop if the episode is terminated
            if terminated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")

    # Plot the rewards per episode
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_episodes + 1), total_rewards, marker="o")
    plt.title("Test Results")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

# Run the test
test_agent(num_episodes=10)
