import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from envs.gym_wrapper import *

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import yaml
import os

import numpy as np
import math
import scipy.stats as st

from moral.active_learning import *
from moral.preference_giver import *
from utils.generate_demos_not_main import *
from utils.evaluate_ppo_not_main import *
from utils.load_config import *
from moral.airl import *
from utils.save_data import *

from drlhp.preference_model import *


def evaluate_pref_model(ppo, pref_model, config, n_eval=1000):
	"""
	:param ppo: Trained policy
	:param config: Environment config
	:param n_eval: Number of evaluation steps
	:return: mean, std of rewards, mean std of pref_model evaluation of actions
	"""
	env = GymWrapper(config.env_id)
	states = env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	obj_logs = []
	obj_returns_traj = []
	obj_returns_action = []
	pref_model_logs = []
	pref_model_returns_traj = []
	pref_model_returns_action = []

	for t in range(n_eval):
		actions, log_probs = ppo.act(states_tensor)
		next_states, reward, done, info = env.step(actions)
		obj_logs.append(reward)
		obj_returns_action.append(obj_logs[-1])
		# print("states = ", states)
		# print("len states = ", len(states))
		pref_model_logs.append(pref_model.evaluate_action(reward).item())
		pref_model_returns_action.append(pref_model_logs[-1])
		
		# for i in len(states):
		#     pref_model_logs.append(pref_model.forward(states[i], next_states[i], config.gamma))

		if done:
			next_states = env.reset()
			obj_logs = np.array(obj_logs).sum(axis=0)
			obj_returns_traj.append(obj_logs)
			obj_logs = []
			pref_model_logs = np.array(pref_model_logs).sum(axis=0)
			pref_model_returns_traj.append(pref_model_logs)
			pref_model_logs = []

		# Prepare state input for next time step
		states = next_states.copy()
		states_tensor = torch.tensor(states).float().to(device)

	obj_returns_traj = np.array(obj_returns_traj)
	obj_means = obj_returns_traj.mean(axis=0)
	obj_std = obj_returns_traj.std(axis=0)

	pref_model_returns_traj = np.array(pref_model_returns_traj)
	pref_model_means = pref_model_returns_traj.mean(axis=0)
	pref_model_std = pref_model_returns_traj.std(axis=0)

	ids = np.argsort(pref_model_returns_traj)[::-1] # reverse
	sorted_returns = obj_returns_traj[ids]
	sorted_pref_rew = pref_model_returns_traj[ids]

	ids = np.argsort(pref_model_returns_action)[::-1] # reverse
	sorted_returns_action = np.array(obj_returns_action)[ids]
	sorted_pref_rew_action = np.array(pref_model_returns_action)[ids]

	# return list(obj_means), list(obj_std), pref_model_means, pref_model_std, sorted_returns, sorted_pref_rew, sorted_returns_action, sorted_pref_rew_action
	return obj_means, sorted_returns, sorted_pref_rew, sorted_returns_action, sorted_pref_rew_action

if __name__ == '__main__':

	# Init WandB & Parameters
	wandb.init(project='test_pref_model', config={
		'env_id': 'randomized_v3',
		'gamma': 0.999,
		'batchsize_pref_modelinator': 512,
		'n_demos': 100,
		# 'preference_model_filename': "generated_data/v3/pref_model/1000q_ParetoDom.pt",
		'preference_model_filename': "generated_data/v3/pref_model/5000q_50b_200e_1>0>2>3.pt",
		# 'expert_agent_filename': "generated_data/v3/moral_agents/[[0, 1, 0, 1], [0, 0, 1, 1]]131_new_norm_v6_v3.pt",
		# 'demos_filename': "generated_data/v3/moral_agents/DEMOS_[[0, 1, 0, 1], [0, 0, 1, 1]]131_new_norm_v6_v3.pk",
		'expert_agent_filename': "generated_data/v3/moral_agents/[[0, 1, 0, 1], [0, 0, 1, 1]]131_norm_v6_v4_div2.pt",
		'demos_filename': "generated_data/v3/moral_agents/DEMOS_[[0, 1, 0, 1], [0, 0, 1, 1]]131_norm_v6_v4_div2.pk",
		'rand_filename': "generated_data/v3/rand/expert.pt",
		'rand_demos_filename': "generated_data/v3/rand/demonstrations.pk",
		'generate_demos': False,
		'env_dim': 4
		})
	config = wandb.config

	# Create Environment
	vec_env = VecEnv(config.env_id, 12)
	states = vec_env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	# Fetch Shapes
	n_actions = vec_env.action_space.n
	obs_shape = vec_env.observation_space.shape
	state_shape = obs_shape[:-1]
	in_channels = obs_shape[-1]


	# experts
	expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	expert_policy.load_state_dict(torch.load(config.expert_agent_filename, map_location=torch.device('cpu')))

	# rand agents
	rand_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	rand_policy.load_state_dict(torch.load(config.rand_filename, map_location=torch.device('cpu')))

	# discriminator
	preference_model = PreferenceModelTEST(config.env_dim).to(device)
	preference_model.load_state_dict(torch.load(config.preference_model_filename, map_location=torch.device('cpu')))

	if config.generate_demos:
		generate_demos_1_expert(config.env_id, config.n_demos, config.expert_agent_filename, config.demos_filename)

	# Load demonstrations & evaluate ppo
	expert_trajectories = pickle.load(open(config.demos_filename, 'rb'))
	mean_rew_obj, sorted_returns, sorted_pref_rew, sorted_returns_action, sorted_pref_rew_action = evaluate_pref_model(expert_policy, preference_model, config)
	print("eval expert : ")
	print(sorted_returns)
	print(sorted_pref_rew)
	print(mean_rew_obj)

	rand_trajectories = pickle.load(open(config.rand_demos_filename, 'rb'))
	mean_rew_obj, sorted_returns, sorted_pref_rew, sorted_returns_action, sorted_pref_rew_action = evaluate_pref_model(rand_policy, preference_model, config)
	print("\n\neval rand : ")
	print(sorted_returns)
	print(sorted_pref_rew)
	print(mean_rew_obj)

	all_combi = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
	all_combi = [np.array(c) for c in all_combi]

	for c in all_combi:
		rew = preference_model.evaluate_action(c).item()
		print(str(c) + " = " + str(rew))