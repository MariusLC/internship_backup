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


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_EVALUATE.yaml"

if __name__ == '__main__':

	c = load_config(CONFIG_PATH, CONFIG_FILENAME)

	# Create Environment
	vec_env = VecEnv(c["env_rad"]+c["env"], 12)
	states = vec_env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	# Fetch Shapes
	n_actions = vec_env.action_space.n
	obs_shape = vec_env.observation_space.shape
	state_shape = obs_shape[:-1]
	in_channels = obs_shape[-1]

	vanilla_path = ""
	if c["vanilla"]:
		vanilla_path = c["vanilla_path"]

	# Init WandB & Parameters
	wandb.init(project='test_discrim', config={
		'env_id': c["env_rad"]+c["env"],
		'gamma': 0.999,
		'batchsize_discriminator': 512,
		#'env_steps': 9e6,
		# 'env_steps': 1000,
		# 'batchsize_ppo': 12,
		# 'n_workers': 12,
		# 'lr_ppo': 3e-4,
		# 'entropy_reg': 0.05,
		# 'lambd': [int(i) for i in args.lambd],
		# 'epsilon': 0.1,
		# 'ppo_epochs': 5
		})
	config = wandb.config


	for i in range(c["nb_experts"]):
		path = c["data_path"]+c["env_path"]+vanilla_path+str(c["experts_weights"][i])+"/"

		expert_filename = path+c["expe_path"]+c["model_ext"]
		rand_filename = c["data_path"]+c["env_path"]+vanilla_path+c["rand_path"]+c["expe_path"]+c["model_ext"]

		demos_filename = path+c["demo_path"]+c["demo_ext"]
		rand_demos_filename = c["data_path"]+c["env_path"]+vanilla_path+c["rand_path"]+c["demo_path"]+c["demo_ext"]
		
		
		# print(demos_filename)

		# experts
		expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
		expert_policy.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))

		# rand agents
		rand_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
		rand_policy.load_state_dict(torch.load(rand_filename, map_location=torch.device('cpu')))


		if c["generate_demos"] :
			generate_demos_1_expert(c["env_rad"]+c["env"], c["nb_demos"], expert_filename, demos_filename)
			# generate_demos_1_expert(c["env_rad"]+c["env"], c["nb_demos"], rand_filename, rand_demos_filename)

		# Load demonstrations & evaluate ppo
		expert_trajectories = pickle.load(open(demos_filename, 'rb'))
		print("eval expert = ", evaluate_ppo(expert_policy, config))
		rand_trajectories = pickle.load(open(rand_demos_filename, 'rb'))
		print("eval rand = ", evaluate_ppo(rand_policy, config))