import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt

import numpy as np

from tester import Test

from network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
# https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146

class PPO:
    def __init__(self, env):
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        #Initialize actor optimizer   
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        
        #Initialize critic optimizer 
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

    def save_model(self, directory, actor_filename="actor.pth", critic_filename="critic.pth"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, actor_filename))
        torch.save(self.critic.state_dict(), os.path.join(directory, critic_filename))
        print(f"Models saved to {directory}")

    def load_model(self, directory, actor_filename="actor.pth", critic_filename="critic.pth"):
        actor_path = os.path.join(directory, actor_filename)
        critic_path = os.path.join(directory, critic_filename)

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"Models loaded from {directory}")
        else:
            print(f"Model files not found in {directory}")

    def _init_hyperparameters(self):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.num_minibatches = 6                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0                               # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold

    # def compute_rtgs(self, batch_rews):
    #     # The rewards-to-go (rtg) per episode per batch to return.
    #     # The shape will be (num timesteps per episode)
    #     batch_rtgs = []
    #     # Iterate through each episode backwards to maintain same order
    #     # in batch_rtgs
    #     for ep_rews in reversed(batch_rews):
    #         discounted_reward = 0 # The discounted reward so far
    #         for rew in reversed(ep_rews):
    #             discounted_reward = rew + discounted_reward * self.gamma
    #             batch_rtgs.insert(0, discounted_reward)
    #     # Convert the rewards-to-go into a tensor
    #     batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    #     return batch_rtgs
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_vals = []
        batch_dones = []
        batch_lens = []            # episodic lengths in batch

        ep_rews = []
        ep_vals = []
        ep_dones = []
        
        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            ep_vals = []
            ep_dones = []

            obs = self.env.reset()
            obs = np.array(obs[0])
            done = False
            steps = 0
            while not done:
                steps =+ 1
                # Increment timesteps ran this batch so far
                t += 1
                ep_dones.append(done)
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
            
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            # Collect episodic length and rewards
            batch_lens.append(steps + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).flatten()

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
    
    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Our computation graph will start later down the line.
        return action.detach().numpy(), log_prob.detach()
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and log probs log_probs
        return V, log_probs, dist.entropy()

    def learn(self, ep_total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        avg_returns = []  # Track average returns per batch
        all_timesteps = []  # Track total timesteps per batch
        return_variances = []  # Track variance of returns per batch
        while t_so_far < ep_total_timesteps:
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
            
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Track average episodic return
            avg_return = batch_rtgs.mean().item()
            variance = np.var([np.sum(ep_rews) for ep_rews in batch_rews])  # Episodic return variance

            avg_returns.append(avg_return)
            all_timesteps.append(t_so_far)
            return_variances.append(variance)

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            for _ in range(self.n_updates_per_iteration):
                frac = (t_so_far - 1.0) /ep_total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                
                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]
                
                    # Calculate V_phi and pi_theta(a_t | s_t)    
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
                    
                    logratios = curr_log_probs - mini_log_prob

                    # Calculate ratios
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                if approx_kl > self.target_kl:
                    break
            
            print(f"Timesteps: {t_so_far}, Average Return: {avg_return}")
        return all_timesteps, avg_returns, return_variances

import gymnasium as gym

import sys
sys.path.append('/home/maranhas/UnBall/rSoccer')
import rsoccer_gym

env = gym.make('VSS-v0', max_episode_steps=1600, render_mode="rgb_array")
model = PPO(env)
# timesteps, avg_returns = model.learn(10000)

# # Save the trained model
# model.save_model("ppo_models")

# To retrain later:
model.load_model("ppo_models_3")
# timesteps, avg_returns, return_variances = model.learn(20000)  # Resume training
# model.save_model("ppo_models_3")

# Visualization
# plt.figure(figsize=(10, 6))
# plt.plot(timesteps, avg_returns, label="PPO Algorithm", color="green", linewidth=2)
# plt.fill_between(timesteps, np.array(avg_returns) - 5, np.array(avg_returns) + 5, color="green", alpha=0.3)  # Fake variance for demo

# plt.title("rSoccer: PPO Performance", fontsize=16)
# plt.xlabel("Total Timesteps", fontsize=12)
# plt.ylabel("Average Episodic Return", fontsize=12)
# plt.legend(loc="lower right", fontsize=10)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()


# Visualization
# plt.figure(figsize=(10, 6))
# std_dev = np.sqrt(return_variances)  # Convert variance to standard deviation
# avg_returns = np.array(avg_returns)
# timesteps = np.array(timesteps)

# plt.plot(timesteps, avg_returns, label="PPO on rSoccer", color="green", linewidth=2)
# plt.fill_between(timesteps, avg_returns - std_dev, avg_returns + std_dev, color="green", alpha=0.3)

# plt.title("rSoccer: PPO Performance", fontsize=16)
# plt.xlabel("Total Timesteps", fontsize=12)
# plt.ylabel("Average Episodic Return", fontsize=12)
# plt.legend(loc="lower right", fontsize=10)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

env = gym.make('VSS-v0', max_episode_steps=1600, render_mode="human")

Test.agent(env, model, num_episodes=10)