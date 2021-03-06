import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PPO(nn.Module):
    def __init__(self, state_shape, in_channels=6, n_actions=9):
        super(PPO, self).__init__()

        # General Parameters
        self.state_shape = state_shape
        self.in_channels = in_channels

        # Network Layers
        self.l1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=2)
        self.l2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2)
        self.actor_l3 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=2)
        self.critic_l3 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=2)
        self.actor_out = nn.Linear(32*(state_shape[0]-3)*(state_shape[1]-3), n_actions)
        self.critic_out = nn.Linear(32*(state_shape[0]-3)*(state_shape[1]-3), 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.state_shape[0], self.state_shape[1])
        x = self.relu(self.l1(x))
        # print("x = ", x.shape)
        x = self.relu(self.l2(x))
        x_actor = self.relu(self.actor_l3(x))
        x_actor = x_actor.view(x_actor.shape[0], -1)
        x_critic = self.relu(self.critic_l3(x))
        x_critic = x_critic.view(x_critic.shape[0], -1)
        x_actor = self.softmax(self.actor_out(x_actor))
        x_critic = self.critic_out(x_critic)

        return x_actor, x_critic

    def act(self, state):
        action_probabilities, _ = self.forward(state)
        m = Categorical(action_probabilities)
        action = m.sample()
        return action.detach().cpu().numpy(), m.log_prob(action).detach().cpu().numpy()

    def evaluate_trajectory(self, tau):
        trajectory_states = torch.tensor(np.array(tau['states'])).float().to(device)
        trajectory_actions = torch.tensor(np.array(tau['actions'])).to(device)
        action_probabilities, critic_values = self.forward(trajectory_states)

        # print(action_probabilities[0][0])
        if math.isnan(action_probabilities[0][0]):
            print("there is a nan value in result of forward in evaluate_trajectory")
            # print(trajectory_actions)
            # print(trajectory_actions.shape)
            # print(trajectory_states)
            # print(trajectory_states.shape)
            for state in tau['states']:
                action_probabilities, critic_values = self.forward(torch.tensor(np.array(state)).float().to(device))
                # if math.isnan(action_probabilities[0][0]):
                    # print("this state gives a action_proba of nan : ")
                    # print(state)

        # print("len(action_probabilities) = ", action_probabilities.shape)
        dist = Categorical(action_probabilities)
        action_entropy = dist.entropy().mean()
        action_log_probabilities = dist.log_prob(trajectory_actions)

        return action_log_probabilities, torch.squeeze(critic_values), action_entropy


class TrajectoryDataset:
    def __init__(self, batch_size, n_workers):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.trajectories = []
        # self.buffer = [{'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'latents': None, 'logs': []}
        #                for i in range(n_workers)]
        self.buffer = [{'states': [], 'actions': [], 'rewards': [], 'advantages': [], 'returns':[], 'log_probs': [], 'latents': None, 'logs': [], 'discounted_rewards':[]}
                       for i in range(n_workers)]
        self.returns_min = math.inf
        self.returns_max = -math.inf
        self.sum = 0
        self.nb_act = 0

    def reset_buffer(self, i):
        # self.buffer[i] = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'latents': None, 'logs': []}
        self.buffer[i] = {'states': [], 'actions': [], 'rewards': [], 'advantages':[], 'returns':[], 'log_probs': [], 'latents': None, 'logs': [], 'discounted_rewards':[]}

    def reset_trajectories(self):
        self.trajectories = []
        self.nb_act = 0
        self.sum = 0
        self.returns_min = math.inf
        self.returns_max = -math.inf

    def write_tuple(self, states, actions, rewards, done, log_probs, logs=None, gamma=0.999):
        # Takes states of shape (n_workers, state_shape[0], state_shape[1])
        for i in range(self.n_workers):
            self.buffer[i]['states'].append(states[i])
            self.buffer[i]['actions'].append(actions[i])
            self.buffer[i]['rewards'].append(rewards[i])
            # self.buffer[i]['discounted_rewards'].append(rewards[i]*(gamma**len(self.buffer[i]['discounted_rewards'])))
            self.buffer[i]['log_probs'].append(log_probs[i])

            if logs is not None:
                self.buffer[i]['logs'].append(logs[i])

            if done[i]:
                self.trajectories.append(self.buffer[i].copy())
                self.nb_act += len(self.buffer[i]['states'])
                self.reset_buffer(i)

        # print("nb traj = ",len(self.trajectories))
        # print("batch_size = ", self.batch_size)
        if len(self.trajectories) >= self.batch_size:
            return True
        else:
            return False

    def write_tuple_2(self, states, actions, rewards, advantages, done, log_probs, logs=None, gamma=0.999):
        # Takes states of shape (n_workers, state_shape[0], state_shape[1])
        for i in range(self.n_workers):
            self.buffer[i]['states'].append(states[i])
            self.buffer[i]['actions'].append(actions[i])
            self.buffer[i]['rewards'].append(rewards[i])
            # self.buffer[i]['discounted_rewards'].append(rewards[i]*(gamma**len(self.buffer[i]['discounted_rewards'])))
            self.buffer[i]['advantages'].append(advantages[i])
            self.buffer[i]['log_probs'].append(log_probs[i])

            if logs is not None:
                self.buffer[i]['logs'].append(logs[i])

            if done[i]:
                self.trajectories.append(self.buffer[i].copy())
                self.reset_buffer(i)

    def write_tuple_3(self, states, actions, rewards, returns, done, log_probs, logs=None, gamma=0.999):
        # Takes states of shape (n_workers, state_shape[0], state_shape[1])
        for i in range(self.n_workers):
            self.buffer[i]['states'].append(states[i])
            self.buffer[i]['actions'].append(actions[i])
            self.buffer[i]['rewards'].append(rewards[i])
            self.buffer[i]['returns'].append(returns[i])
            # self.buffer[i]['discounted_rewards'].append(rewards[i]*(gamma**len(self.buffer[i]['discounted_rewards'])))
            self.buffer[i]['log_probs'].append(log_probs[i])

            # print("returns[i] = ", returns[i])
            self.returns_min = min(self.returns_min, returns[i])
            self.returns_max = max(self.returns_max, returns[i])
            self.sum += returns[i]
            self.nb_act += 1

            if logs is not None:
                self.buffer[i]['logs'].append(logs[i])

            if done[i]:
                self.trajectories.append(self.buffer[i].copy())
                # self.nb_act += len(self.buffer[i]['states'])
                self.reset_buffer(i)

        if len(self.trajectories) >= self.batch_size:
            return True
        else:
            return False

    def log_rewards(self):
        # Calculates (undiscounted) returns in self.trajectories
        returns = [0 for i in range(len(self.trajectories))]
        for i, tau in enumerate(self.trajectories):
            returns[i] = sum(tau['rewards'])
        return returns

    def log_advantages(self):
        # Calculates (undiscounted) returns in self.trajectories
        returns = [0 for i in range(len(self.trajectories))]
        for i, tau in enumerate(self.trajectories):
            returns[i] = sum(tau['advantages'])
        return returns

    def log_returns(self):
        # Calculates (undiscounted) returns in self.trajectories
        returns = [0 for i in range(len(self.trajectories))]
        for i, tau in enumerate(self.trajectories):
            returns[i] = sum(tau['returns'])
        return returns

    def log_objectives(self):
        # Calculates achieved objectives objectives in self.trajectories
        objective_logs = []
        for i, tau in enumerate(self.trajectories):
            objective_logs.append(list(np.array(tau['logs']).sum(axis=0)))

        return np.array(objective_logs)
    

    def normalize_v1(self, value):
        print("self.returns_max = ", self.returns_max)
        print("self.returns_min = ", self.returns_min)
        normalization_v1 = (value - self.returns_min)/(self.returns_max - self.returns_min)
        return normalization_v1

    def normalize_v2(self, value):
        normalization_v1 = self.normalization_v1(value)
        mean = self.sum / self.nb_act
        normalization_v2 = normalization_v1/abs(mean)
        return normalization_v2

    def normalize_v3(self, value):
        returns = 0
        for tau in self.trajectories:
            returns += sum(tau['returns'])
        mean_over_1_traj = returns/len(self.trajectories)
        normalization_v3 = value/abs(mean_over_1_traj)

        # mean_over_1_traj = self.sum / self.nb_act
        # normalization_v2 = normalization_v1/abs(mean)

        return normalization_v3

        


def g_clip(epsilon, A):
    return torch.tensor([1 + epsilon if i else 1 - epsilon for i in A >= 0]).to(device) * A


def update_policy(ppo, dataset, optimizer, gamma, epsilon, n_epochs, entropy_reg):
    for epoch in range(n_epochs):
        batch_loss = 0
        value_loss = 0
        for i, tau in enumerate(dataset.trajectories):
            reward_togo = 0
            returns = []
            # rewards are scalarized rewards (discrim.forward * w_posterior[i])
            normalized_reward = np.array(tau['rewards'])
            normalized_reward = (normalized_reward - normalized_reward.mean())/(normalized_reward.std()+1e-5)
            for r in normalized_reward[::-1]:
                # Compute rewards-to-go and advantage estimates
                reward_togo = r + gamma * reward_togo
                returns.insert(0, reward_togo)
            action_log_probabilities, critic_values, action_entropy = ppo.evaluate_trajectory(tau)
            advantages = torch.tensor(np.array(returns)).to(device) - critic_values.detach().to(device)
            likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(np.array(tau['log_probs'])).detach().to(device))
            clipped_losses = -torch.min(likelihood_ratios * advantages, g_clip(epsilon, advantages))
            batch_loss += torch.mean(clipped_losses) - entropy_reg * action_entropy
            value_loss += torch.mean((torch.tensor(np.array(returns)).to(device) - critic_values) ** 2)
        overall_loss = (batch_loss + value_loss) / dataset.batch_size
        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()


def update_policy_v2(ppo, dataset, optimizer, gamma, epsilon, n_epochs, entropy_reg, target_kl=0.01):
    for epoch in range(n_epochs):
        for i, tau in enumerate(dataset.trajectories):
            # print("i = ", i)
            reward_togo = 0
            returns = []
            normalized_reward = np.array(tau['rewards'])
            normalized_reward = (normalized_reward - normalized_reward.mean())/(normalized_reward.std()+1e-5)
            for r in normalized_reward[::-1]:
                # Compute rewards-to-go and advantage estimates
                reward_togo = r + gamma * reward_togo
                returns.insert(0, reward_togo)
            action_log_probabilities, critic_values, action_entropy = ppo.evaluate_trajectory(tau)

            kl = (torch.tensor(np.array(tau['log_probs'])).detach().to(device) - action_log_probabilities).mean()
            # print("kl = ", kl)
            if kl > 1.5 * target_kl:
                # print('Early stopping at step %d due to reaching max kl.'%i)
                break

            advantages = torch.tensor(returns).to(device) - critic_values.detach().to(device)
            likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(np.array(tau['log_probs'])).detach().to(device))
            clipped_losses = -torch.min(likelihood_ratios * advantages, g_clip(epsilon, advantages))
            batch_loss = torch.mean(clipped_losses) - entropy_reg * action_entropy
            value_loss = torch.mean((torch.tensor(np.array(returns)).to(device) - critic_values) ** 2)


            overall_loss = batch_loss + value_loss
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()


def update_policy_v3(ppo, dataset, optimizer, gamma, epsilon, n_epochs, entropy_reg, wandb, target_kl=0.01):
    for epoch in range(n_epochs):
        batch_loss = 0
        value_loss = 0
        batch_loss_2 = 0
        value_loss_2 = 0
        kl = 0
        for i, tau in enumerate(dataset.trajectories):
            # print("i = ", i)
            reward_togo = 0
            returns = []
            normalized_reward = np.array(tau['rewards'])
            normalized_reward = (normalized_reward - normalized_reward.mean())/(normalized_reward.std()+1e-5)
            for r in normalized_reward[::-1]:
                # Compute rewards-to-go and advantage estimates
                reward_togo = r + gamma * reward_togo
                returns.insert(0, reward_togo)
            action_log_probabilities, critic_values, action_entropy = ppo.evaluate_trajectory(tau)
            advantages = torch.tensor(returns).to(device) - critic_values.detach().to(device)
            likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(np.array(tau['log_probs'])).detach().to(device))
            clipped_losses = -torch.min(likelihood_ratios * advantages, g_clip(epsilon, advantages))
            batch_loss += clipped_losses.sum()
            value_loss += ((torch.tensor(np.array(returns)).to(device) - critic_values) ** 2).sum()
            kl += (torch.tensor(np.array(tau['log_probs'])).detach().to(device) - action_log_probabilities).sum()

            batch_loss_2 += torch.mean(clipped_losses) - entropy_reg * action_entropy
            value_loss_2 += torch.mean((torch.tensor(np.array(returns)).to(device) - critic_values) ** 2)

        kl = kl / dataset.nb_act
        # print(dataset.batch_size)
        # print(dataset.nb_act)
        # print("kl = ", kl)
        if kl > 1.5 * target_kl:
            print('Early stopping at step %d due to reaching max kl.'%i)
            break

        overall_loss = (batch_loss + value_loss - entropy_reg * action_entropy) / dataset.nb_act
        overall_loss_2 = (batch_loss_2 + value_loss_2) / dataset.batch_size

        # print("overall_loss = ", overall_loss)
        # print("overall_loss_2 = ", overall_loss_2)
        wandb.log({'overall_loss': overall_loss})
        wandb.log({'overall_loss_2': overall_loss_2})

        optimizer.zero_grad()
        overall_loss_2.backward()
        optimizer.step()
