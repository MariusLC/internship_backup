import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math
import wandb
from envs.gym_wrapper import *

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
        self.buffer = [{'states': [], 'actions': [], 'rewards': [], 'airl_rewards': [], 'returns':[], 'vectorized_rewards': [], 'log_probs': [], 'latents': None, 'logs': [], 'discounted_rewards':[]}
                       for i in range(n_workers)]
        self.sum = 0
        self.nb_act = 0
        self.utopia_point = None
        self.returns_min_traj = math.inf
        self.returns_max_traj = -math.inf

        # calculated with an expert in the non ethical objective, previously learned during a ppo phase.
        self.utopia_point_expert = None

    def reset_buffer(self, i):
        # self.buffer[i] = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'latents': None, 'logs': []}
        self.buffer[i] = {'states': [], 'actions': [], 'rewards': [], 'airl_rewards':[], 'returns':[], 'vectorized_rewards': [], 'log_probs': [], 'latents': None, 'logs': [], 'discounted_rewards':[]}

    def reset_trajectories(self):
        self.trajectories = []
        self.nb_act = 0
        self.sum = 0
        self.utopia_point = None

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

    def write_tuple_norm(self, states, actions, rewards, returns, airl_rewards, done, log_probs, logs=None, gamma=0.999):
        # Takes states of shape (n_workers, state_shape[0], state_shape[1])
        # print(returns)
        for i in range(self.n_workers):
            self.buffer[i]['states'].append(states[i])
            self.buffer[i]['actions'].append(actions[i])
            self.buffer[i]['airl_rewards'].append(airl_rewards[i]) # il faut probablement probablement switch les axes
            self.buffer[i]['returns'].append(returns[i])
            self.buffer[i]['log_probs'].append(log_probs[i])

            # HOF non eth params for normalization
            # self.returns_min = min(self.returns_min, returns[i][0])
            # self.returns_max = max(self.returns_max, returns[i][0])
            self.sum += returns[i][0]
            self.nb_act += 1

            if logs is not None:
                self.buffer[i]['logs'].append(logs[i])

            if done[i]:
                self.trajectories.append(self.buffer[i].copy())
                self.reset_buffer(i)
                returns_arr = np.array(self.trajectories[-1]['returns'])
                # print("sum traj returns = ", returns_arr)
                # print("sum traj returns = ", returns_arr.sum(axis=0))
                self.returns_min_traj = min(self.returns_min_traj, returns_arr.sum(axis=0)[0])
                self.returns_max_traj = max(self.returns_max_traj, returns_arr.sum(axis=0)[0])

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


        

    def log_vectorized_rew_sum(self):
        returns = [0 for i in range(len(self.trajectories))]
        for i, tau in enumerate(self.trajectories):
            # print("tau['vectorized_rewards'] = ", tau['vectorized_rewards'])
            # print("sum = ", sum(tau['vectorized_rewards']))
            returns[i] = sum(tau['vectorized_rewards'])
        # print("returns = ", returns)
        return returns

    def log_returns_sum(self):
        # Calculates (undiscounted) returns in self.trajectories
        returns = [0 for i in range(len(self.trajectories))]
        for i, tau in enumerate(self.trajectories):
            returns[i] = sum(tau['returns'])
        return returns
    

    def normalize_v1(self, value, traj_size):
        normalization_v1 = (value - self.returns_min_traj/traj_size)/(self.returns_max_traj - self.returns_min_traj)
        return normalization_v1

    def normalize_v2(self, value, traj_size):
        normalization_v2 = value/abs(self.utopia_point)
        return normalization_v2

    def normalize_v3(self, value, traj_size):
        normalization_v3 = value/abs(self.utopia_point_expert)
        return normalization_v3

    def normalize_v4(self, value, traj_size):
        normalization_v4 = value/abs(self.utopia_point_expert)
        return normalization_v4

    def estimate_utopia_point(self, expert_policy, env_id, steps=10000):
        env = GymWrapper(env_id)
        states = env.reset()
        states_tensor = torch.tensor(states).float().to(device)

        # Fetch Shapes
        n_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        state_shape = obs_shape[:-1]
        in_channels = obs_shape[-1]

        # Init returns
        estimated_returns = []
        running_returns = 0

        for t in range(steps):
            actions, log_probs = expert_policy.act(states_tensor)
            next_states, rewards, done, info = env.step(actions)
            curr_reward = rewards[0]

            if done:
                curr_reward = 0
                next_states = env.reset()
            running_returns += curr_reward

            if done:
                estimated_returns.append(running_returns)
                running_returns = 0

            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        # l'utopia point est simplement la moyenne des rewards estimés par le discriminateur des trajectoires finies sur n pas de temps,
        # en se référant à l'imitation policy pour le choix des actions
        self.utopia_point_expert = sum(estimated_returns)/len(estimated_returns)
        print("self.utopia_point_expert = ", self.utopia_point_expert)

        return self.utopia_point_expert

    def estimate_normalisation_points(self, non_eth_norm, expert_policy, env_id, steps=10000):
        if non_eth_norm == "v3":
            self.estimate_utopia_point(expert_policy, env_id, steps)


    def compute_scalarized_rewards(self, w_posterior_mean, non_eth_norm, wandb):
        if non_eth_norm == "v0": # pas de normalisation de l'obj non ethique (comme dans MORAL de base)
            non_eth_norm_fct = None
        else:
            if non_eth_norm == "v1": # normalisation classique par rapport aux valeurs min et max all time sur une traj (value - min)/(max - min)
                non_eth_norm_fct = self.normalize_v1
            elif non_eth_norm == "v2": # division par la moyenne des rewards sur une trajectoire pour tout le batch de données courant (data_set)
                non_eth_norm_fct = self.normalize_v2
            elif non_eth_norm == "v3": # division par la moyenne des rewards sur une trajectoire pour tout le batch de données courant (data_set)
                non_eth_norm_fct = self.normalize_v3
            elif non_eth_norm == "v4": # division par la moyenne des rewards sur une trajectoire pour tout le batch de données courant (data_set)
                non_eth_norm_fct = self.normalize_v4
            self.compute_utopia()
            self.compute_normalization_non_eth(non_eth_norm_fct)

        mean_vectorized_rewards = [0 for i in range(len(self.trajectories[0]["airl_rewards"][0])+1)]
        for i in range(len(self.trajectories)):
            mean_vectorized_rewards_1_traj = [0 for i in range(len(self.trajectories[0]["airl_rewards"][0])+1)]
            for j in range(len(self.trajectories[i]["states"])):
                # print("ret = ", self.trajectories[i]["returns"][j][0])
                # print("rew_airl = ", self.trajectories[i]["airl_rewards"][j])
                # print("vector = ", np.concatenate(([self.trajectories[i]["returns"][j][0]], self.trajectories[i]["airl_rewards"][j])))
                self.trajectories[i]["vectorized_rewards"].append(np.concatenate(([self.trajectories[i]["returns"][j][0]], self.trajectories[i]["airl_rewards"][j]))) # np array ?
                # print("w_posterior_mean = ", w_posterior_mean)
                # print("self.trajectories[i] = ", self.trajectories[i]["vectorized_rewards"][j])
                self.trajectories[i]["rewards"].append(np.dot(w_posterior_mean, self.trajectories[i]["vectorized_rewards"][j]))
                # self.log_wandb(self.trajectories[i]["vectorized_rewards"][-1], self.trajectories[i]["airl_rewards"][j], wandb, w_posterior_mean)
                mean_vectorized_rewards_1_traj += self.trajectories[i]["vectorized_rewards"][-1]
                # mean_scalarized_rewards += self.trajectories[i]["rewards"][-1]
            # print("r0 traj = ", np.array(self.trajectories[i]["returns"])[:,0])
            # print("r0 traj = ", np.array(self.trajectories[i]["returns"]).sum(axis=0)[0])
            # if wandb != None :
            #     self.log_wandb_1_traj(mean_vectorized_rewards_1_traj, wandb, w_posterior_mean)
            mean_vectorized_rewards += mean_vectorized_rewards_1_traj
        mean_vectorized_rewards = mean_vectorized_rewards/len(self.trajectories)
        print("mean_vectorized_rewards = ", mean_vectorized_rewards)

        return mean_vectorized_rewards


    def compute_normalization_non_eth(self, non_eth_norm):
        # print("self.returns_max_traj = ", self.returns_max_traj)
        # print("self.returns_min_traj = ", self.returns_min_traj)
        for i in range(len(self.trajectories)):
            for j in range(len(self.trajectories[i]["states"])):
                traj_size = len(self.trajectories[i]["states"])
                # self.trajectories[i][j]["returns"] = non_eth_norm(self.trajectories[i][j]["returns"])
                # print(self.trajectories[i])
                # print(self.trajectories[i]["returns"])
                # print(self.trajectories[i]["returns"][j])
                # self.trajectories[i]["returns"][j] = self.normalize_v1(self.trajectories[i]["returns"][j],traj_size)
                self.trajectories[i]["returns"][j] = non_eth_norm(self.trajectories[i]["returns"][j],traj_size)
                # print("2 = ", self.trajectories[i]["returns"][j])

    def compute_utopia(self):
        # print("self.sum = ", self.sum)
        # print("nb traj = ", len(self.trajectories))
        self.utopia_point = self.sum / len(self.trajectories)
        # print("utopia_point = ", self.utopia_point)

    # def log_wandb(self, vectorized_rewards, rewards, wandb, w_posterior_mean):
    #     # print("vectorized_rewards 1 traj = ", vectorized_rewards)
    #     # mean_rew = np.array(vectorized_rewards).mean(axis=0) # ?
    #     # returns_vb, rewards_vb = volume_buffer.get_data()
    #     # rewards_vb = np.array(rewards)
    #     # rewards_vb = rewards_vb.mean(axis=0) # sum over trajectories
    #     # rewards_vb = rewards_vb.mean(axis=0) # sum over workers ?
    #     # print(rewards_vb)                  # we get the mean rewards over all actions in the buffer
    #     for i in range(len(vectorized_rewards)):
    #         wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]})
    #         wandb.log({'vectorized_rew_mean ['+str(i)+']': vectorized_rewards[i]})
    #         wandb.log({'weighted_rew_mean ['+str(i)+']': w_posterior_mean[i] * vectorized_rewards[i]})
    #         # wandb.log({'rewards_mean ['+str(i)+']': rewards_vb[i]})
    #         # print('w_posterior_mean ['+str(i)+']'+ str(w_posterior_mean[i]))
    #         # print('vectorized_rew_mean ['+str(i)+']'+ str(vectorized_rewards[i]))
    #         # print('weighted_rew_mean ['+str(i)+']'+ str(w_posterior_mean[i] * vectorized_rewards[i]))
    #         # print('rewards_mean ['+str(i)+']'+ str(rewards_vb[i]))

    # def log_wandb_1_traj(self, vectorized_rewards, wandb, w_posterior_mean):
    #     # print("vectorized_rewards = ", vectorized_rewards)
    #     for i in range(len(vectorized_rewards)):
    #         wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]})
    #         wandb.log({'vectorized_rew_mean ['+str(i)+']': vectorized_rewards[i]})
    #         wandb.log({'weighted_rew_mean ['+str(i)+']': w_posterior_mean[i] * vectorized_rewards[i]})
    #         # print('w_posterior_mean ['+str(i)+']'+ str(w_posterior_mean[i]))
    #         # print('vectorized_rew_mean ['+str(i)+']'+ str(vectorized_rewards[i]))
    #         # print('weighted_rew_mean ['+str(i)+']'+ str(w_posterior_mean[i] * vectorized_rewards[i]))


        


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
            # print("reward_togo = ",reward_togo)
            action_log_probabilities, critic_values, action_entropy = ppo.evaluate_trajectory(tau)
            advantages = torch.tensor(np.array(returns)).to(device) - critic_values.detach().to(device)
            likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(np.array(tau['log_probs'])).detach().to(device))
            clipped_losses = -torch.min(likelihood_ratios * advantages, g_clip(epsilon, advantages))
            # print("clipped_losses_mean = ", torch.mean(clipped_losses) )
            # print("entropy_reg = ", entropy_reg)
            # print("action_entropy = ", action_entropy)
            batch_loss += torch.mean(clipped_losses) - entropy_reg * action_entropy
            value_loss += torch.mean((torch.tensor(np.array(returns)).to(device) - critic_values) ** 2)
            # print("batch_loss = ", torch.mean(clipped_losses) - entropy_reg * action_entropy)
            # print("value_loss = ", torch.mean((torch.tensor(np.array(returns)).to(device) - critic_values) ** 2))
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
        # wandb.log({'overall_loss': overall_loss})
        # wandb.log({'overall_loss_2': overall_loss_2})

        optimizer.zero_grad()
        overall_loss_2.backward()
        optimizer.step()
