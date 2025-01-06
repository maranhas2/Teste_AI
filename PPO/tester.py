import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Test():
    def agent(env, model, num_episodes=10):
        """
        Test the trained PPO agent in the given environment.

        Parameters:
        - env: The environment to test the agent in.
        - model: The trained PPO model.
        - num_episodes: Number of episodes to test the agent for.
        - render: Whether to render the environment during testing.

        Returns:
        - avg_reward: Average reward over all test episodes.
        - avg_length: Average episode length over all test episodes.
        """
        total_rewards = []
        episode_lengths = []

        for ep in range(num_episodes):
            obs = env.reset()
            obs = np.array(obs[0])
            done = False
            ep_reward = 0
            steps = 0

            while not done:
                # Get action from the trained model
                action, _ = model.get_action(torch.tensor(obs, dtype=torch.float))
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = np.array(obs)
                done = terminated or truncated

                ep_reward += reward
                steps += 1

            total_rewards.append(ep_reward)
            episode_lengths.append(steps)

            print(f"Episode {ep + 1}/{num_episodes}: Reward: {ep_reward:.2f}, Length: {steps}")

        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)

        print("\nTest Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.2f}")

        return avg_reward, avg_length
